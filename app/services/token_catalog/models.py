"""
Token Catalog Models

Data models for enriched token metadata and portfolio profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


class MarketCapTier(str, Enum):
    """Market cap tiers for token classification."""
    MEGA = "mega"      # > $10B
    LARGE = "large"    # $1B - $10B
    MID = "mid"        # $100M - $1B
    SMALL = "small"    # $10M - $100M
    MICRO = "micro"    # < $10M


class TokenSector(str, Enum):
    """Top-level sector classification."""
    DEFI = "DeFi"
    INFRASTRUCTURE = "Infrastructure"
    GAMING = "Gaming"
    SOCIAL = "Social"
    AI_DATA = "AI/Data"
    MEME = "Meme"
    UNKNOWN = "Unknown"


class TokenSubsector(str, Enum):
    """Subsector classification within sectors."""
    # DeFi
    DEX = "DEX"
    LENDING = "Lending"
    DERIVATIVES = "Derivatives"
    YIELD = "Yield"
    STABLECOIN = "Stablecoin"
    INSURANCE = "Insurance"

    # Infrastructure
    L1 = "L1"
    L2 = "L2"
    ORACLE = "Oracle"
    BRIDGE = "Bridge"
    STORAGE = "Storage"
    PRIVACY = "Privacy"

    # Gaming
    GAMING = "Gaming"
    METAVERSE = "Metaverse"
    NFT_INFRA = "NFT Infrastructure"

    # Social
    SOCIAL_NETWORK = "Social Network"
    CREATOR = "Creator"

    # AI/Data
    AI = "AI"
    DATA_INDEX = "Data/Index"

    # Meme
    MEME_TOKEN = "Meme Token"

    # Other
    GOVERNANCE = "Governance"
    WRAPPED = "Wrapped"
    OTHER = "Other"


@dataclass
class RelatedToken:
    """Represents a related token for correlation analysis."""
    address: str
    chain_id: int
    relationship: str  # "same_project", "competitor", "derivative", "wrapped"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "chainId": self.chain_id,
            "relationship": self.relationship,
        }


@dataclass
class TokenTaxonomy:
    """Token classification/taxonomy data."""
    categories: List[str] = field(default_factory=list)
    sector: Optional[TokenSector] = None
    subsector: Optional[TokenSubsector] = None
    is_stablecoin: bool = False
    is_wrapped: bool = False
    is_lp_token: bool = False
    is_governance_token: bool = False
    is_native: bool = False
    market_cap_tier: Optional[MarketCapTier] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categories": self.categories,
            "sector": self.sector.value if self.sector else None,
            "subsector": self.subsector.value if self.subsector else None,
            "isStablecoin": self.is_stablecoin,
            "isWrapped": self.is_wrapped,
            "isLpToken": self.is_lp_token,
            "isGovernanceToken": self.is_governance_token,
            "isNative": self.is_native,
            "marketCapTier": self.market_cap_tier.value if self.market_cap_tier else None,
        }


@dataclass
class EnrichedToken:
    """Fully enriched token with metadata and taxonomy."""
    # Core identity
    address: str
    chain_id: int
    symbol: str
    name: str
    decimals: int

    # Taxonomy
    taxonomy: TokenTaxonomy = field(default_factory=TokenTaxonomy)

    # Project/Protocol info
    project_name: Optional[str] = None
    project_slug: Optional[str] = None
    coingecko_id: Optional[str] = None
    defillama_id: Optional[str] = None

    # Links
    logo_url: Optional[str] = None
    website: Optional[str] = None
    twitter: Optional[str] = None
    discord: Optional[str] = None
    github: Optional[str] = None

    # Related tokens
    related_tokens: List[RelatedToken] = field(default_factory=list)

    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Data freshness
    last_updated: Optional[datetime] = None
    data_source: str = "unknown"
    enrichment_version: int = 1

    @property
    def canonical_id(self) -> str:
        """Unique identifier: chain_id:address."""
        return f"{self.chain_id}:{self.address.lower()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Convex storage."""
        return {
            "address": self.address.lower(),
            "chainId": self.chain_id,
            "symbol": self.symbol,
            "name": self.name,
            "decimals": self.decimals,
            "categories": self.taxonomy.categories,
            "sector": self.taxonomy.sector.value if self.taxonomy.sector else None,
            "subsector": self.taxonomy.subsector.value if self.taxonomy.subsector else None,
            "projectName": self.project_name,
            "projectSlug": self.project_slug,
            "coingeckoId": self.coingecko_id,
            "defillamaId": self.defillama_id,
            "logoUrl": self.logo_url,
            "website": self.website,
            "twitter": self.twitter,
            "discord": self.discord,
            "github": self.github,
            "marketCapTier": self.taxonomy.market_cap_tier.value if self.taxonomy.market_cap_tier else None,
            "isStablecoin": self.taxonomy.is_stablecoin,
            "isWrapped": self.taxonomy.is_wrapped,
            "isLpToken": self.taxonomy.is_lp_token,
            "isGovernanceToken": self.taxonomy.is_governance_token,
            "isNative": self.taxonomy.is_native,
            "relatedTokens": [r.to_dict() for r in self.related_tokens],
            "dataSource": self.data_source,
            "enrichmentVersion": self.enrichment_version,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrichedToken":
        """Create from Convex data."""
        taxonomy = TokenTaxonomy(
            categories=data.get("categories", []),
            sector=TokenSector(data["sector"]) if data.get("sector") else None,
            subsector=TokenSubsector(data["subsector"]) if data.get("subsector") else None,
            is_stablecoin=data.get("isStablecoin", False),
            is_wrapped=data.get("isWrapped", False),
            is_lp_token=data.get("isLpToken", False),
            is_governance_token=data.get("isGovernanceToken", False),
            is_native=data.get("isNative", False),
            market_cap_tier=MarketCapTier(data["marketCapTier"]) if data.get("marketCapTier") else None,
        )

        related = [
            RelatedToken(
                address=r["address"],
                chain_id=r["chainId"],
                relationship=r["relationship"],
            )
            for r in data.get("relatedTokens", [])
        ]

        return cls(
            address=data["address"],
            chain_id=data["chainId"],
            symbol=data["symbol"],
            name=data["name"],
            decimals=data["decimals"],
            taxonomy=taxonomy,
            project_name=data.get("projectName"),
            project_slug=data.get("projectSlug"),
            coingecko_id=data.get("coingeckoId"),
            defillama_id=data.get("defillamaId"),
            logo_url=data.get("logoUrl"),
            website=data.get("website"),
            twitter=data.get("twitter"),
            discord=data.get("discord"),
            github=data.get("github"),
            related_tokens=related,
            description=data.get("description"),
            tags=data.get("tags", []),
            data_source=data.get("dataSource", "unknown"),
            enrichment_version=data.get("enrichmentVersion", 1),
        )


@dataclass
class SectorAllocation:
    """Portfolio sector allocation."""
    sector: TokenSector
    percentage: float
    value_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector": self.sector.value,
            "percentage": self.percentage,
            "valueUsd": self.value_usd,
        }


@dataclass
class RiskProfile:
    """Portfolio risk assessment."""
    diversification_score: float  # 0-100
    stablecoin_percent: float
    meme_percent: float
    concentration_risk: float  # Top holding percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diversificationScore": self.diversification_score,
            "stablecoinPercent": self.stablecoin_percent,
            "memePercent": self.meme_percent,
            "concentrationRisk": self.concentration_risk,
        }


@dataclass
class PortfolioProfile:
    """Complete portfolio analysis profile."""
    wallet_address: str

    # Allocation analysis
    sector_allocation: Dict[str, float]  # sector -> percentage
    category_exposure: Dict[str, float]  # category -> percentage

    # Risk assessment
    risk_profile: RiskProfile

    # Token distribution
    tokens_by_tier: Dict[str, int]  # tier -> count

    # Metadata
    total_value_usd: Optional[float] = None
    token_count: int = 0
    calculated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "walletAddress": self.wallet_address,
            "sectorAllocation": self.sector_allocation,
            "categoryExposure": self.category_exposure,
            "riskProfile": self.risk_profile.to_dict(),
            "tokensByTier": self.tokens_by_tier,
            "portfolioValueUsd": self.total_value_usd,
            "tokenCount": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioProfile":
        """Create from Convex data."""
        risk_data = data.get("riskProfile", {})
        return cls(
            wallet_address=data["walletAddress"],
            sector_allocation=data.get("sectorAllocation", {}),
            category_exposure=data.get("categoryExposure", {}),
            risk_profile=RiskProfile(
                diversification_score=risk_data.get("diversificationScore", 0),
                stablecoin_percent=risk_data.get("stablecoinPercent", 0),
                meme_percent=risk_data.get("memePercent", 0),
                concentration_risk=risk_data.get("concentrationRisk", 0),
            ),
            tokens_by_tier=data.get("tokensByTier", {}),
            total_value_usd=data.get("portfolioValueUsd"),
            token_count=len(data.get("tokens", [])),
        )
