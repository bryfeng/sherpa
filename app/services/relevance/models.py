"""
Relevance Scoring Models

Data structures for portfolio-aware content relevance scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal


class RelevanceFactor(str, Enum):
    """Factors that contribute to relevance scoring."""
    DIRECT_HOLDING = "direct_holding"      # User holds the mentioned token
    SAME_PROJECT = "same_project"          # Same project/protocol family
    SECTOR_MATCH = "sector_match"          # Same sector (DeFi, Gaming, etc.)
    SUBSECTOR_MATCH = "subsector_match"    # Same subsector (DEX, Lending, etc.)
    COMPETITOR = "competitor"              # Competitor to held tokens
    CORRELATION = "correlation"            # Correlated assets (ETH affects L2s)
    POSITION_WEIGHT = "position_weight"    # Larger positions = more relevant
    CATEGORY_OVERLAP = "category_overlap"  # Shared categories


# Default weights for each factor
DEFAULT_WEIGHTS: Dict[RelevanceFactor, float] = {
    RelevanceFactor.DIRECT_HOLDING: 0.40,
    RelevanceFactor.SAME_PROJECT: 0.20,
    RelevanceFactor.SECTOR_MATCH: 0.10,
    RelevanceFactor.SUBSECTOR_MATCH: 0.05,
    RelevanceFactor.COMPETITOR: 0.08,
    RelevanceFactor.CORRELATION: 0.07,
    RelevanceFactor.POSITION_WEIGHT: 0.05,
    RelevanceFactor.CATEGORY_OVERLAP: 0.05,
}


@dataclass
class TokenHolding:
    """A token in the user's portfolio."""
    symbol: str
    address: Optional[str] = None
    chain_id: int = 1
    value_usd: Decimal = Decimal("0")
    percentage: float = 0.0  # Percentage of portfolio

    # Enriched data (from token catalog)
    sector: Optional[str] = None
    subsector: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    project_slug: Optional[str] = None
    related_tokens: List[str] = field(default_factory=list)  # Related token symbols

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        if isinstance(self.value_usd, (int, float)):
            self.value_usd = Decimal(str(self.value_usd))


@dataclass
class PortfolioContext:
    """
    User's portfolio context for relevance scoring.

    Pre-computed from token catalog enrichment.
    """
    holdings: List[TokenHolding]

    # Aggregated data for fast lookups
    symbols: Set[str] = field(default_factory=set)
    addresses: Set[str] = field(default_factory=set)
    sectors: Dict[str, float] = field(default_factory=dict)  # sector -> weight
    subsectors: Dict[str, float] = field(default_factory=dict)
    categories: Dict[str, float] = field(default_factory=dict)
    projects: Set[str] = field(default_factory=set)

    # Total portfolio value
    total_value_usd: Decimal = Decimal("0")

    def __post_init__(self):
        """Build lookup structures from holdings."""
        if not self.symbols:
            self._build_lookups()

    def _build_lookups(self):
        """Build fast lookup structures from holdings."""
        self.symbols = set()
        self.addresses = set()
        self.projects = set()
        self.total_value_usd = Decimal("0")  # Reset before summing
        sector_values: Dict[str, Decimal] = {}
        subsector_values: Dict[str, Decimal] = {}
        category_values: Dict[str, Decimal] = {}

        for holding in self.holdings:
            self.symbols.add(holding.symbol.upper())

            if holding.address:
                self.addresses.add(holding.address.lower())

            if holding.project_slug:
                self.projects.add(holding.project_slug.lower())

            # Aggregate sector values
            if holding.sector:
                sector_values[holding.sector] = sector_values.get(
                    holding.sector, Decimal("0")
                ) + holding.value_usd

            if holding.subsector:
                subsector_values[holding.subsector] = subsector_values.get(
                    holding.subsector, Decimal("0")
                ) + holding.value_usd

            for cat in holding.categories:
                category_values[cat] = category_values.get(
                    cat, Decimal("0")
                ) + holding.value_usd

            self.total_value_usd += holding.value_usd

        # Convert to percentages
        if self.total_value_usd > 0:
            self.sectors = {
                k: float(v / self.total_value_usd)
                for k, v in sector_values.items()
            }
            self.subsectors = {
                k: float(v / self.total_value_usd)
                for k, v in subsector_values.items()
            }
            self.categories = {
                k: float(v / self.total_value_usd)
                for k, v in category_values.items()
            }

    def get_holding(self, symbol: str) -> Optional[TokenHolding]:
        """Get holding by symbol."""
        symbol_upper = symbol.upper()
        for holding in self.holdings:
            if holding.symbol == symbol_upper:
                return holding
        return None

    def has_symbol(self, symbol: str) -> bool:
        """Check if portfolio contains a symbol."""
        return symbol.upper() in self.symbols

    def has_sector(self, sector: str) -> bool:
        """Check if portfolio has exposure to a sector."""
        return sector in self.sectors

    def get_sector_weight(self, sector: str) -> float:
        """Get portfolio weight in a sector (0-1)."""
        return self.sectors.get(sector, 0.0)


@dataclass
class ContentContext:
    """
    Context about a piece of content (news, alert) for relevance scoring.

    Extracted from processed news items.
    """
    # Token references
    tokens: List[str] = field(default_factory=list)  # Symbols
    token_relevance: Dict[str, float] = field(default_factory=dict)  # symbol -> relevance

    # Classification
    sectors: List[str] = field(default_factory=list)
    subsectors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Projects mentioned
    projects: List[str] = field(default_factory=list)

    # Content metadata
    importance: float = 0.5  # 0-1
    sentiment: float = 0.0   # -1 to 1

    def __post_init__(self):
        self.tokens = [t.upper() for t in self.tokens]

    @classmethod
    def from_processed_news(cls, news: Dict[str, Any]) -> "ContentContext":
        """Create from a processed news item."""
        related_tokens = news.get("relatedTokens", [])

        return cls(
            tokens=[t.get("symbol", "").upper() for t in related_tokens if t.get("symbol")],
            token_relevance={
                t.get("symbol", "").upper(): t.get("relevanceScore", 0.5)
                for t in related_tokens
                if t.get("symbol")
            },
            sectors=news.get("relatedSectors", []),
            subsectors=[],  # Extracted from categories if needed
            categories=news.get("relatedCategories", []),
            projects=[],  # Could be extracted from token catalog
            importance=news.get("importance", {}).get("score", 0.5) if isinstance(news.get("importance"), dict) else 0.5,
            sentiment=news.get("sentiment", {}).get("score", 0.0) if isinstance(news.get("sentiment"), dict) else 0.0,
        )


@dataclass
class RelevanceBreakdown:
    """Detailed breakdown of relevance score by factor."""
    factor: RelevanceFactor
    score: float  # 0-1 contribution before weighting
    weight: float  # Weight applied
    weighted_score: float  # score * weight
    details: str = ""  # Human-readable explanation
    matched_items: List[str] = field(default_factory=list)  # What matched

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor": self.factor.value,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "weightedScore": round(self.weighted_score, 4),
            "details": self.details,
            "matchedItems": self.matched_items,
        }


@dataclass
class RelevanceScore:
    """
    Final relevance score with breakdown.

    Score ranges from 0 (not relevant) to 1 (highly relevant).
    """
    score: float  # Final weighted score (0-1)
    breakdown: List[RelevanceBreakdown]
    explanation: str  # Human-readable summary

    # Thresholds for categorization
    HIGH_RELEVANCE = 0.7
    MEDIUM_RELEVANCE = 0.4
    LOW_RELEVANCE = 0.2

    @property
    def level(self) -> str:
        """Get relevance level as string."""
        if self.score >= self.HIGH_RELEVANCE:
            return "high"
        elif self.score >= self.MEDIUM_RELEVANCE:
            return "medium"
        elif self.score >= self.LOW_RELEVANCE:
            return "low"
        else:
            return "minimal"

    @property
    def is_relevant(self) -> bool:
        """Check if content meets minimum relevance threshold."""
        return self.score >= self.LOW_RELEVANCE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "isRelevant": self.is_relevant,
            "explanation": self.explanation,
            "breakdown": [b.to_dict() for b in self.breakdown],
        }

    def get_top_factors(self, n: int = 3) -> List[RelevanceBreakdown]:
        """Get top N contributing factors."""
        sorted_breakdown = sorted(
            self.breakdown,
            key=lambda x: x.weighted_score,
            reverse=True
        )
        return sorted_breakdown[:n]


# Known competitor relationships
COMPETITOR_MAP: Dict[str, List[str]] = {
    # DEXes
    "UNI": ["SUSHI", "CRV", "BAL", "1INCH"],
    "SUSHI": ["UNI", "CRV", "BAL"],
    "CRV": ["UNI", "BAL", "SUSHI"],

    # Lending
    "AAVE": ["COMP", "MKR", "MORPHO"],
    "COMP": ["AAVE", "MKR"],
    "MKR": ["AAVE", "COMP"],

    # L2s
    "ARB": ["OP", "MATIC", "ZK"],
    "OP": ["ARB", "MATIC", "ZK"],
    "MATIC": ["ARB", "OP"],

    # Oracles
    "LINK": ["PYTH", "BAND", "API3"],
    "PYTH": ["LINK", "BAND"],

    # Liquid Staking
    "LDO": ["RPL", "CBETH", "FRAX"],
    "RPL": ["LDO"],

    # Perps
    "GMX": ["DYDX", "SNX"],
    "DYDX": ["GMX", "SNX"],
}

# Correlation groups (assets that move together)
CORRELATION_GROUPS: Dict[str, List[str]] = {
    # ETH ecosystem
    "ETH": ["WETH", "STETH", "RETH", "CBETH", "ARB", "OP", "MATIC"],

    # BTC ecosystem
    "BTC": ["WBTC", "TBTC", "RENBTC"],

    # Stablecoins
    "USDC": ["USDT", "DAI", "FRAX", "LUSD"],
    "USDT": ["USDC", "DAI"],

    # DeFi blue chips
    "UNI": ["AAVE", "CRV", "MKR", "COMP"],
    "AAVE": ["UNI", "CRV", "MKR", "COMP"],

    # SOL ecosystem
    "SOL": ["RAY", "ORCA", "JUP", "BONK"],
}
