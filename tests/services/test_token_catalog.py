"""
Tests for Token Catalog Service

Tests for token enrichment, classification, and portfolio analysis.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from app.services.token_catalog import (
    TokenCatalogService,
    EnrichedToken,
    TokenTaxonomy,
    PortfolioProfile,
    RiskProfile,
    TokenSector,
    TokenSubsector,
    MarketCapTier,
    RelatedToken,
    TokenClassifier,
    CoinGeckoSource,
    DefiLlamaSource,
)
from app.services.token_catalog.service import TokenHolding
from app.services.token_catalog.sources import CoinGeckoTokenInfo, DefiLlamaProtocol


# =============================================================================
# Token Classifier Tests
# =============================================================================

class TestTokenClassifier:
    """Tests for token classification logic."""

    @pytest.fixture
    def classifier(self):
        return TokenClassifier()

    def test_classify_stablecoin_by_symbol(self, classifier):
        """Test stablecoin detection by symbol."""
        taxonomy = classifier.classify("USDC", "USD Coin")
        assert taxonomy.is_stablecoin is True
        assert taxonomy.sector == TokenSector.DEFI
        assert taxonomy.subsector == TokenSubsector.STABLECOIN
        assert "stablecoin" in taxonomy.categories

    def test_classify_stablecoin_by_name(self, classifier):
        """Test stablecoin detection by name."""
        taxonomy = classifier.classify("XYZ", "Some Dollar Stablecoin")
        assert taxonomy.is_stablecoin is True

    def test_classify_wrapped_token(self, classifier):
        """Test wrapped token detection."""
        taxonomy = classifier.classify("WETH", "Wrapped Ether")
        assert taxonomy.is_wrapped is True
        assert "wrapped" in taxonomy.categories

    def test_classify_lp_token(self, classifier):
        """Test LP token detection."""
        taxonomy = classifier.classify("UNI-V2", "Uniswap V2 Pool Token")
        assert taxonomy.is_lp_token is True
        assert taxonomy.sector == TokenSector.DEFI
        assert taxonomy.subsector == TokenSubsector.DEX

    def test_classify_native_token(self, classifier):
        """Test native token detection."""
        taxonomy = classifier.classify("ETH", "Ethereum")
        assert taxonomy.is_native is True
        assert taxonomy.sector == TokenSector.INFRASTRUCTURE
        assert taxonomy.subsector == TokenSubsector.L1
        assert "native" in taxonomy.categories

    def test_classify_l2_token(self, classifier):
        """Test L2 token detection."""
        taxonomy = classifier.classify("ARB", "Arbitrum")
        assert taxonomy.sector == TokenSector.INFRASTRUCTURE
        assert taxonomy.subsector == TokenSubsector.L2

    def test_classify_dex_token(self, classifier):
        """Test DEX governance token detection."""
        taxonomy = classifier.classify("UNI", "Uniswap")
        assert taxonomy.sector == TokenSector.DEFI
        assert taxonomy.subsector == TokenSubsector.DEX
        assert taxonomy.is_governance_token is True

    def test_classify_lending_token(self, classifier):
        """Test lending protocol token detection."""
        taxonomy = classifier.classify("AAVE", "Aave Token")
        assert taxonomy.sector == TokenSector.DEFI
        assert taxonomy.subsector == TokenSubsector.LENDING
        assert taxonomy.is_governance_token is True

    def test_classify_oracle_token(self, classifier):
        """Test oracle token detection."""
        taxonomy = classifier.classify("LINK", "Chainlink")
        assert taxonomy.sector == TokenSector.INFRASTRUCTURE
        assert taxonomy.subsector == TokenSubsector.ORACLE

    def test_classify_meme_token(self, classifier):
        """Test meme token detection."""
        taxonomy = classifier.classify("DOGE", "Dogecoin")
        assert taxonomy.sector == TokenSector.MEME
        assert taxonomy.subsector == TokenSubsector.MEME_TOKEN
        assert "meme" in taxonomy.categories

    def test_classify_ai_token(self, classifier):
        """Test AI token detection."""
        taxonomy = classifier.classify("FET", "Fetch.ai")
        assert taxonomy.sector == TokenSector.AI_DATA
        assert taxonomy.subsector == TokenSubsector.AI

    def test_classify_unknown_token(self, classifier):
        """Test unknown token classification."""
        taxonomy = classifier.classify("RANDOM", "Random Token")
        assert taxonomy.sector == TokenSector.UNKNOWN

    def test_market_cap_tier_mega(self, classifier):
        """Test mega cap tier classification."""
        taxonomy = classifier.classify(
            "BTC", "Bitcoin",
            market_cap=500_000_000_000,  # $500B
        )
        assert taxonomy.market_cap_tier == MarketCapTier.MEGA

    def test_market_cap_tier_large(self, classifier):
        """Test large cap tier classification."""
        taxonomy = classifier.classify(
            "XYZ", "Large Token",
            market_cap=5_000_000_000,  # $5B
        )
        assert taxonomy.market_cap_tier == MarketCapTier.LARGE

    def test_market_cap_tier_mid(self, classifier):
        """Test mid cap tier classification."""
        taxonomy = classifier.classify(
            "XYZ", "Mid Token",
            market_cap=500_000_000,  # $500M
        )
        assert taxonomy.market_cap_tier == MarketCapTier.MID

    def test_market_cap_tier_small(self, classifier):
        """Test small cap tier classification."""
        taxonomy = classifier.classify(
            "XYZ", "Small Token",
            market_cap=50_000_000,  # $50M
        )
        assert taxonomy.market_cap_tier == MarketCapTier.SMALL

    def test_market_cap_tier_micro(self, classifier):
        """Test micro cap tier classification."""
        taxonomy = classifier.classify(
            "XYZ", "Micro Token",
            market_cap=5_000_000,  # $5M
        )
        assert taxonomy.market_cap_tier == MarketCapTier.MICRO

    def test_coingecko_categories_applied(self, classifier):
        """Test CoinGecko category mapping."""
        taxonomy = classifier.classify(
            "XYZ", "Some Token",
            coingecko_categories=["decentralized-exchange", "governance"],
        )
        assert taxonomy.sector == TokenSector.DEFI
        assert taxonomy.subsector == TokenSubsector.DEX
        assert "defi" in taxonomy.categories
        assert "dex" in taxonomy.categories

    def test_find_related_tokens(self, classifier):
        """Test finding related tokens."""
        related = classifier.find_related_tokens("ETH", chain_id=1)

        # ETH should have wrapped and staking derivatives
        assert len(related) > 0
        relationships = [r.relationship for r in related]
        assert "wrapped" in relationships or "derivative" in relationships

    def test_find_competitor_tokens(self, classifier):
        """Test finding competitor tokens."""
        related = classifier.find_related_tokens("AAVE", chain_id=1)

        relationships = [r.relationship for r in related]
        assert "competitor" in relationships


# =============================================================================
# Enriched Token Model Tests
# =============================================================================

class TestEnrichedToken:
    """Tests for EnrichedToken model."""

    def test_canonical_id(self):
        """Test canonical ID generation."""
        token = EnrichedToken(
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            chain_id=1,
            symbol="USDC",
            name="USD Coin",
            decimals=6,
        )

        assert token.canonical_id == "1:0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def test_to_dict(self):
        """Test serialization to dict."""
        token = EnrichedToken(
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            chain_id=1,
            symbol="USDC",
            name="USD Coin",
            decimals=6,
            taxonomy=TokenTaxonomy(
                categories=["stablecoin"],
                sector=TokenSector.DEFI,
                subsector=TokenSubsector.STABLECOIN,
                is_stablecoin=True,
            ),
        )

        data = token.to_dict()

        assert data["symbol"] == "USDC"
        assert data["chainId"] == 1
        assert data["isStablecoin"] is True
        assert data["sector"] == "DeFi"
        assert "stablecoin" in data["categories"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "chainId": 1,
            "symbol": "USDC",
            "name": "USD Coin",
            "decimals": 6,
            "categories": ["stablecoin"],
            "sector": "DeFi",
            "subsector": "Stablecoin",
            "isStablecoin": True,
            "isWrapped": False,
            "isLpToken": False,
            "isGovernanceToken": False,
            "isNative": False,
            "relatedTokens": [],
            "tags": [],
        }

        token = EnrichedToken.from_dict(data)

        assert token.symbol == "USDC"
        assert token.chain_id == 1
        assert token.taxonomy.is_stablecoin is True
        assert token.taxonomy.sector == TokenSector.DEFI


# =============================================================================
# Portfolio Profile Tests
# =============================================================================

class TestPortfolioProfile:
    """Tests for portfolio profile analysis."""

    @pytest.fixture
    def mock_holdings(self):
        """Sample holdings for testing."""
        return [
            TokenHolding(
                address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                chain_id=1,
                symbol="USDC",
                name="USD Coin",
                decimals=6,
                balance=10000.0,
                value_usd=10000.0,
            ),
            TokenHolding(
                address="0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
                chain_id=1,
                symbol="AAVE",
                name="Aave Token",
                decimals=18,
                balance=50.0,
                value_usd=5000.0,
            ),
            TokenHolding(
                address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                chain_id=1,
                symbol="UNI",
                name="Uniswap",
                decimals=18,
                balance=100.0,
                value_usd=3000.0,
            ),
        ]

    def test_risk_profile_stablecoin_percent(self):
        """Test stablecoin percentage calculation."""
        profile = RiskProfile(
            diversification_score=50.0,
            stablecoin_percent=55.56,  # 10000 / 18000
            meme_percent=0.0,
            concentration_risk=55.56,  # Top holding
        )

        assert profile.stablecoin_percent == 55.56

    def test_risk_profile_to_dict(self):
        """Test risk profile serialization."""
        profile = RiskProfile(
            diversification_score=75.0,
            stablecoin_percent=25.0,
            meme_percent=5.0,
            concentration_risk=30.0,
        )

        data = profile.to_dict()

        assert data["diversificationScore"] == 75.0
        assert data["stablecoinPercent"] == 25.0
        assert data["memePercent"] == 5.0
        assert data["concentrationRisk"] == 30.0

    def test_portfolio_profile_to_dict(self):
        """Test portfolio profile serialization."""
        profile = PortfolioProfile(
            wallet_address="0x1234567890123456789012345678901234567890",
            sector_allocation={"DeFi": 80.0, "Infrastructure": 20.0},
            category_exposure={"lending": 30.0, "dex": 20.0, "stablecoin": 50.0},
            risk_profile=RiskProfile(
                diversification_score=60.0,
                stablecoin_percent=50.0,
                meme_percent=0.0,
                concentration_risk=40.0,
            ),
            tokens_by_tier={"mega": 1, "large": 2, "mid": 0, "small": 0, "micro": 0},
            total_value_usd=18000.0,
            token_count=3,
        )

        data = profile.to_dict()

        assert data["walletAddress"].startswith("0x")
        assert "DeFi" in data["sectorAllocation"]
        assert data["tokenCount"] == 3


# =============================================================================
# CoinGecko Source Tests
# =============================================================================

class TestCoinGeckoSource:
    """Tests for CoinGecko data source."""

    @pytest.fixture
    def source(self):
        return CoinGeckoSource()

    def test_map_categories_to_taxonomy(self, source):
        """Test category mapping."""
        taxonomy = source.map_categories_to_taxonomy([
            "Decentralized Exchange",
            "Governance",
        ])

        assert taxonomy.sector == TokenSector.DEFI
        assert taxonomy.subsector == TokenSubsector.DEX
        assert "defi" in taxonomy.categories
        assert "dex" in taxonomy.categories

    def test_map_stablecoin_category(self, source):
        """Test stablecoin category mapping."""
        taxonomy = source.map_categories_to_taxonomy(["Stablecoins"])

        assert taxonomy.is_stablecoin is True

    def test_calculate_market_cap_tier(self, source):
        """Test market cap tier calculation."""
        assert source.calculate_market_cap_tier(15_000_000_000) == MarketCapTier.MEGA
        assert source.calculate_market_cap_tier(5_000_000_000) == MarketCapTier.LARGE
        assert source.calculate_market_cap_tier(500_000_000) == MarketCapTier.MID
        assert source.calculate_market_cap_tier(50_000_000) == MarketCapTier.SMALL
        assert source.calculate_market_cap_tier(5_000_000) == MarketCapTier.MICRO
        assert source.calculate_market_cap_tier(None) is None


# =============================================================================
# DefiLlama Source Tests
# =============================================================================

class TestDefiLlamaSource:
    """Tests for DefiLlama data source."""

    @pytest.fixture
    def source(self):
        return DefiLlamaSource()

    def test_map_category_dex(self, source):
        """Test DEX category mapping."""
        sector, subsector = source.map_category_to_taxonomy("Dexes")
        assert sector == TokenSector.DEFI
        assert subsector == TokenSubsector.DEX

    def test_map_category_lending(self, source):
        """Test lending category mapping."""
        sector, subsector = source.map_category_to_taxonomy("Lending")
        assert sector == TokenSector.DEFI
        assert subsector == TokenSubsector.LENDING

    def test_map_category_bridge(self, source):
        """Test bridge category mapping."""
        sector, subsector = source.map_category_to_taxonomy("Bridge")
        assert sector == TokenSector.INFRASTRUCTURE
        assert subsector == TokenSubsector.BRIDGE

    def test_map_unknown_category(self, source):
        """Test unknown category mapping."""
        sector, subsector = source.map_category_to_taxonomy("Unknown Category")
        assert sector is None
        assert subsector is None


# =============================================================================
# Token Catalog Service Tests
# =============================================================================

class TestTokenCatalogService:
    """Tests for the main token catalog service."""

    @pytest.fixture
    def service(self):
        return TokenCatalogService()

    @pytest.mark.asyncio
    async def test_enrich_token_creates_base(self, service):
        """Test basic token enrichment without external data."""
        # Patch external calls
        with patch.object(service._coingecko, 'get_token_by_contract', return_value=None):
            with patch.object(service._defillama, 'find_protocol_for_token', return_value=None):
                token = await service.enrich_token(
                    address="0x1234567890123456789012345678901234567890",
                    chain_id=1,
                    symbol="TEST",
                    name="Test Token",
                    decimals=18,
                )

                assert token.symbol == "TEST"
                assert token.chain_id == 1
                assert token.address == "0x1234567890123456789012345678901234567890"

    @pytest.mark.asyncio
    async def test_enrich_token_applies_coingecko(self, service):
        """Test token enrichment with CoinGecko data."""
        mock_info = CoinGeckoTokenInfo(
            id="uniswap",
            symbol="UNI",
            name="Uniswap",
            categories=["Decentralized Exchange", "Governance"],
            market_cap=5_000_000_000,
            image="https://example.com/uni.png",
            description="Uniswap protocol governance token",
            links={"homepage": ["https://uniswap.org"]},
            platforms={},
        )

        with patch.object(service._coingecko, 'get_token_by_contract', return_value=mock_info):
            with patch.object(service._defillama, 'find_protocol_for_token', return_value=None):
                token = await service.enrich_token(
                    address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                    chain_id=1,
                )

                assert token.coingecko_id == "uniswap"
                assert token.logo_url == "https://example.com/uni.png"
                assert token.taxonomy.market_cap_tier == MarketCapTier.LARGE

    @pytest.mark.asyncio
    async def test_portfolio_profile_calculation(self, service):
        """Test portfolio profile calculation."""
        holdings = [
            TokenHolding(
                address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                chain_id=1,
                symbol="USDC",
                name="USD Coin",
                decimals=6,
                balance=10000.0,
                value_usd=10000.0,
            ),
            TokenHolding(
                address="0xdead000000000000000000000000000000000000",
                chain_id=1,
                symbol="DOGE",
                name="Dogecoin",
                decimals=8,
                balance=100000.0,
                value_usd=5000.0,
            ),
        ]

        # Mock the token enrichment
        async def mock_get_token(address, chain_id, enrich_if_missing=True):
            if "usdc" in address.lower() or "a0b8" in address.lower():
                return EnrichedToken(
                    address=address,
                    chain_id=chain_id,
                    symbol="USDC",
                    name="USD Coin",
                    decimals=6,
                    taxonomy=TokenTaxonomy(
                        categories=["stablecoin"],
                        sector=TokenSector.DEFI,
                        is_stablecoin=True,
                    ),
                )
            else:
                return EnrichedToken(
                    address=address,
                    chain_id=chain_id,
                    symbol="DOGE",
                    name="Dogecoin",
                    decimals=8,
                    taxonomy=TokenTaxonomy(
                        categories=["meme"],
                        sector=TokenSector.MEME,
                    ),
                )

        with patch.object(service, 'get_token', side_effect=mock_get_token):
            profile = await service.get_portfolio_profile(
                holdings,
                wallet_address="0x1234567890123456789012345678901234567890",
            )

            # Verify sector allocation
            assert "DeFi" in profile.sector_allocation
            assert "Meme" in profile.sector_allocation

            # Verify risk profile
            assert profile.risk_profile.stablecoin_percent > 0
            assert profile.risk_profile.meme_percent > 0
            assert profile.token_count == 2

    @pytest.mark.asyncio
    async def test_empty_portfolio_profile(self, service):
        """Test empty portfolio profile."""
        profile = await service.get_portfolio_profile(
            [],
            wallet_address="0x1234567890123456789012345678901234567890",
        )

        assert profile.token_count == 0
        assert profile.total_value_usd == 0
        assert profile.risk_profile.diversification_score == 0


# =============================================================================
# Integration Tests (with mocked external APIs)
# =============================================================================

class TestTokenCatalogIntegration:
    """Integration tests for the token catalog service."""

    @pytest.mark.asyncio
    async def test_full_enrichment_flow(self):
        """Test the full token enrichment flow."""
        service = TokenCatalogService()

        try:
            # Mock both external sources
            mock_cg_info = CoinGeckoTokenInfo(
                id="aave",
                symbol="AAVE",
                name="Aave",
                categories=["Lending/Borrowing", "Governance"],
                market_cap=2_000_000_000,
                image="https://example.com/aave.png",
                description="Aave is a decentralized lending protocol",
                links={"homepage": ["https://aave.com"], "twitter_screen_name": "aaborrowAave"},
                platforms={"ethereum": "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9"},
            )

            mock_dl_protocol = DefiLlamaProtocol(
                id="aave",
                name="Aave",
                slug="aave",
                category="Lending",
                chains=["Ethereum", "Polygon", "Arbitrum"],
                tvl=10_000_000_000,
                symbol="AAVE",
                twitter="AaveAave",
                url="https://aave.com",
                gecko_id="aave",
                address=None,
            )

            with patch.object(service._coingecko, 'get_token_by_contract', return_value=mock_cg_info):
                with patch.object(service._defillama, 'find_protocol_for_token', return_value=mock_dl_protocol):
                    token = await service.enrich_token(
                        address="0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
                        chain_id=1,
                    )

                    # Verify enrichment
                    assert token.symbol == "AAVE"
                    assert token.coingecko_id == "aave"
                    assert token.defillama_id == "aave"
                    assert token.project_name == "Aave"
                    assert token.taxonomy.sector == TokenSector.DEFI
                    assert token.taxonomy.subsector == TokenSubsector.LENDING
                    assert token.taxonomy.is_governance_token is True

        finally:
            await service.close()
