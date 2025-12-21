"""
Tests for the TokenResolutionService.

Covers:
- EVM token resolution (symbol, alias, address)
- Solana token resolution (symbol, alias, mint address)
- Chain detection from query
- Confidence scoring
- Ambiguity detection
- Portfolio context
- Async and sync resolution paths
- Mock provider integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from typing import List, Optional

from app.services.token_resolution import (
    TokenResolutionService,
    TokenMatch,
    ResolutionSource,
    AmbiguityResult,
    get_token_resolution_service,
    TOKEN_REGISTRY,
    SOLANA_TOKEN_REGISTRY,
    CONFIDENCE_EXACT_ADDRESS,
    CONFIDENCE_REGISTRY_EXACT,
    CONFIDENCE_REGISTRY_ALIAS,
    CONFIDENCE_PORTFOLIO,
    CONFIDENCE_JUPITER_EXACT,
    CONFIDENCE_JUPITER_SEARCH,
    CONFIDENCE_COINGECKO_EXACT,
    AMBIGUITY_THRESHOLD,
    SOLANA_NATIVE_MINT,
)
from app.core.chain_types import ChainId, SOLANA_CHAIN_ID, is_solana_chain, is_evm_chain_id


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def service():
    """Create a TokenResolutionService without providers."""
    return TokenResolutionService()


@pytest.fixture
def mock_jupiter_token():
    """Create a mock JupiterToken dataclass."""
    @dataclass
    class MockJupiterToken:
        address: str
        symbol: str
        name: str
        decimals: int
        logo_uri: Optional[str] = None
        tags: List[str] = None
        coingecko_id: Optional[str] = None
    return MockJupiterToken


@pytest.fixture
def mock_jupiter_provider(mock_jupiter_token):
    """Create a mock Jupiter provider."""
    provider = MagicMock()
    provider.get_token_by_mint = AsyncMock(return_value=None)
    provider.search_by_symbol = AsyncMock(return_value=[])
    return provider


@pytest.fixture
def mock_coingecko_provider():
    """Create a mock CoinGecko provider."""
    provider = MagicMock()
    provider.get_token_info = AsyncMock(return_value=None)
    provider.search_coins = AsyncMock(return_value=[])
    return provider


@pytest.fixture
def service_with_providers(mock_coingecko_provider, mock_jupiter_provider):
    """Create a TokenResolutionService with mock providers."""
    return TokenResolutionService(
        coingecko_provider=mock_coingecko_provider,
        jupiter_provider=mock_jupiter_provider,
    )


# =============================================================================
# Chain Type Tests
# =============================================================================

class TestChainTypes:
    """Test chain type detection and helpers."""

    def test_solana_chain_id_is_string(self):
        assert SOLANA_CHAIN_ID == "solana"
        assert isinstance(SOLANA_CHAIN_ID, str)

    def test_is_solana_chain(self):
        assert is_solana_chain("solana") is True
        assert is_solana_chain(1) is False
        assert is_solana_chain(137) is False

    def test_is_evm_chain_id(self):
        assert is_evm_chain_id(1) is True
        assert is_evm_chain_id(137) is True
        assert is_evm_chain_id(8453) is True
        assert is_evm_chain_id("solana") is False


# =============================================================================
# TokenMatch Tests
# =============================================================================

class TestTokenMatch:
    """Test TokenMatch dataclass properties."""

    def test_evm_token_match_properties(self):
        match = TokenMatch(
            chain_id=1,
            address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            symbol="USDC",
            name="USD Coin",
            decimals=6,
            confidence=0.95,
            source=ResolutionSource.REGISTRY,
        )
        assert match.is_evm is True
        assert match.is_solana is False
        assert match.canonical_id == "1:0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def test_solana_token_match_properties(self):
        match = TokenMatch(
            chain_id=SOLANA_CHAIN_ID,
            address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            symbol="USDC",
            name="USD Coin",
            decimals=6,
            confidence=0.95,
            source=ResolutionSource.REGISTRY,
        )
        assert match.is_solana is True
        assert match.is_evm is False
        # Solana addresses are case-sensitive (Base58)
        assert match.canonical_id == "solana:EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    def test_to_dict_includes_all_fields(self):
        match = TokenMatch(
            chain_id=1,
            address="0xtest",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            confidence=0.9,
            source=ResolutionSource.REGISTRY,
            is_native=False,
            coingecko_id="test-token",
            tags=["defi", "stablecoin"],
        )
        d = match.to_dict()
        assert d["chain_id"] == 1
        assert d["symbol"] == "TEST"
        assert d["is_solana"] is False
        assert d["tags"] == ["defi", "stablecoin"]


# =============================================================================
# EVM Token Resolution Tests (Sync)
# =============================================================================

class TestEVMResolutionSync:
    """Test synchronous EVM token resolution."""

    def test_resolve_eth_by_symbol(self, service):
        matches = service.resolve_sync("ETH", chain_id=1)
        assert len(matches) >= 1
        eth = matches[0]
        assert eth.symbol == "ETH"
        assert eth.chain_id == 1
        assert eth.is_native is True
        assert eth.confidence == CONFIDENCE_REGISTRY_EXACT

    def test_resolve_usdc_by_symbol_multiple_chains(self, service):
        # USDC exists on multiple chains
        matches = service.resolve_sync("USDC")
        assert len(matches) >= 3  # At least Ethereum, Base, Arbitrum
        symbols = {m.symbol for m in matches}
        assert "USDC" in symbols

    def test_resolve_usdc_specific_chain(self, service):
        matches = service.resolve_sync("USDC", chain_id=1)
        assert len(matches) == 1
        assert matches[0].chain_id == 1
        assert matches[0].address == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def test_resolve_by_alias(self, service):
        # "wrapped eth" is an alias for WETH
        matches = service.resolve_sync("wrapped eth", chain_id=1)
        assert len(matches) >= 1
        assert matches[0].symbol == "WETH"
        assert matches[0].confidence == CONFIDENCE_REGISTRY_ALIAS

    def test_resolve_by_address(self, service):
        # USDC address on Ethereum
        usdc_addr = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        matches = service.resolve_sync(usdc_addr)
        assert len(matches) >= 1
        assert matches[0].symbol == "USDC"
        assert matches[0].confidence == CONFIDENCE_EXACT_ADDRESS

    def test_resolve_unknown_token_returns_empty(self, service):
        matches = service.resolve_sync("NONEXISTENT_TOKEN_XYZ")
        assert matches == []

    def test_case_insensitive_resolution(self, service):
        matches_lower = service.resolve_sync("eth", chain_id=1)
        matches_upper = service.resolve_sync("ETH", chain_id=1)
        matches_mixed = service.resolve_sync("Eth", chain_id=1)
        assert matches_lower[0].symbol == matches_upper[0].symbol == matches_mixed[0].symbol


# =============================================================================
# Solana Token Resolution Tests (Sync)
# =============================================================================

class TestSolanaResolutionSync:
    """Test synchronous Solana token resolution."""

    def test_resolve_sol_by_symbol(self, service):
        matches = service.resolve_sync("SOL", chain_id=SOLANA_CHAIN_ID)
        assert len(matches) >= 1
        sol = matches[0]
        assert sol.symbol == "SOL"
        assert sol.chain_id == SOLANA_CHAIN_ID
        assert sol.is_native is True
        assert sol.decimals == 9

    def test_resolve_bonk_by_symbol(self, service):
        matches = service.resolve_sync("BONK", chain_id=SOLANA_CHAIN_ID)
        assert len(matches) >= 1
        bonk = matches[0]
        assert bonk.symbol == "BONK"
        assert bonk.chain_id == SOLANA_CHAIN_ID
        assert bonk.address == "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"

    def test_resolve_solana_by_mint_address(self, service):
        # BONK mint address
        bonk_mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
        matches = service.resolve_sync(bonk_mint)
        assert len(matches) >= 1
        assert matches[0].symbol == "BONK"
        assert matches[0].confidence == CONFIDENCE_EXACT_ADDRESS

    def test_resolve_solana_usdc_by_symbol(self, service):
        matches = service.resolve_sync("USDC", chain_id=SOLANA_CHAIN_ID)
        assert len(matches) >= 1
        usdc = matches[0]
        assert usdc.symbol == "USDC"
        assert usdc.chain_id == SOLANA_CHAIN_ID
        assert usdc.address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    def test_resolve_by_solana_alias(self, service):
        # "jupiter" is an alias for JUP
        matches = service.resolve_sync("jupiter", chain_id=SOLANA_CHAIN_ID)
        assert len(matches) >= 1
        assert matches[0].symbol == "JUP"

    def test_include_solana_false_excludes_solana_results(self, service):
        # SOL should not appear when include_solana=False
        matches = service.resolve_sync("SOL", include_solana=False)
        for m in matches:
            assert m.chain_id != SOLANA_CHAIN_ID


# =============================================================================
# Cross-Chain Resolution Tests
# =============================================================================

class TestCrossChainResolution:
    """Test resolution across both EVM and Solana."""

    def test_usdc_resolves_on_both_chains(self, service):
        # Use larger top_k to get results from all chains (5 EVM + 1 Solana)
        matches = service.resolve_sync("USDC", top_k=10)
        chains = {m.chain_id for m in matches}
        assert 1 in chains  # Ethereum
        assert SOLANA_CHAIN_ID in chains  # Solana

    def test_chain_detection_from_evm_address(self, service):
        # EVM address format should not match Solana
        evm_addr = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        matches = service.resolve_sync(evm_addr)
        for m in matches:
            assert is_evm_chain_id(m.chain_id)

    def test_chain_detection_from_solana_address(self, service):
        # Solana mint address should only match Solana
        solana_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        matches = service.resolve_sync(solana_mint)
        assert len(matches) >= 1
        assert all(m.chain_id == SOLANA_CHAIN_ID for m in matches)

    def test_sol_symbol_auto_detects_solana(self, service):
        # "SOL" uniquely identifies Solana chain
        matches = service.resolve_sync("SOL")
        # Should primarily return Solana SOL
        assert matches[0].chain_id == SOLANA_CHAIN_ID


# =============================================================================
# Confidence and Ambiguity Tests
# =============================================================================

class TestConfidenceAndAmbiguity:
    """Test confidence scoring and ambiguity detection."""

    def test_exact_address_highest_confidence(self, service):
        usdc_addr = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        matches = service.resolve_sync(usdc_addr, chain_id=1)
        assert matches[0].confidence == CONFIDENCE_EXACT_ADDRESS

    def test_registry_exact_confidence(self, service):
        matches = service.resolve_sync("ETH", chain_id=1)
        assert matches[0].confidence == CONFIDENCE_REGISTRY_EXACT

    def test_alias_lower_confidence_than_exact(self, service):
        exact_matches = service.resolve_sync("WETH", chain_id=1)
        alias_matches = service.resolve_sync("wrapped eth", chain_id=1)
        assert exact_matches[0].confidence > alias_matches[0].confidence

    def test_is_ambiguous_with_no_matches(self, service):
        matches = service.resolve_sync("NONEXISTENT_XYZ")
        assert service.is_ambiguous(matches) is True

    def test_is_ambiguous_with_low_confidence(self, service):
        # Create artificial low-confidence match
        match = TokenMatch(
            chain_id=1,
            address="0x123",
            symbol="LOW",
            name="Low Confidence",
            decimals=18,
            confidence=0.3,
            source=ResolutionSource.FUZZY,
        )
        assert service.is_ambiguous([match]) is True

    def test_is_ambiguous_with_multiple_equal_confidence(self, service):
        match1 = TokenMatch(
            chain_id=1, address="0x1", symbol="A", name="Token A",
            decimals=18, confidence=0.9, source=ResolutionSource.REGISTRY,
        )
        match2 = TokenMatch(
            chain_id=137, address="0x2", symbol="A", name="Token A",
            decimals=18, confidence=0.9, source=ResolutionSource.REGISTRY,
        )
        assert service.is_ambiguous([match1, match2]) is True

    def test_not_ambiguous_with_clear_winner(self, service):
        match1 = TokenMatch(
            chain_id=1, address="0x1", symbol="A", name="Token A",
            decimals=18, confidence=1.0, source=ResolutionSource.EXACT_ADDRESS,
        )
        match2 = TokenMatch(
            chain_id=137, address="0x2", symbol="A", name="Token A",
            decimals=18, confidence=0.6, source=ResolutionSource.COINGECKO,
        )
        assert service.is_ambiguous([match1, match2]) is False

    def test_get_ambiguity_result(self, service):
        matches = service.resolve_sync("USDC")  # Multiple chains
        result = service.get_ambiguity_result("USDC", matches)
        assert isinstance(result, AmbiguityResult)
        assert result.query == "USDC"
        assert "USDC" in result.options_text

    def test_get_best_match_returns_none_when_ambiguous(self, service):
        matches = service.resolve_sync("USDC")  # Multiple chains with same confidence
        if service.is_ambiguous(matches):
            assert service.get_best_match(matches) is None

    def test_get_best_match_returns_top_when_clear(self, service):
        # Single chain lookup should not be ambiguous
        matches = service.resolve_sync("ETH", chain_id=1)
        best = service.get_best_match(matches)
        assert best is not None
        assert best.symbol == "ETH"


# =============================================================================
# Portfolio Context Tests
# =============================================================================

class TestPortfolioContext:
    """Test portfolio-aware resolution."""

    def test_portfolio_provides_context(self, service):
        portfolio = {
            "tokens": [
                {
                    "chain_id": 1,
                    "symbol": "CUSTOM",
                    "name": "Custom Token",
                    "address": "0xcustom123",
                    "decimals": 18,
                }
            ]
        }
        matches = service.resolve_sync("CUSTOM", portfolio=portfolio)
        assert len(matches) >= 1
        assert matches[0].symbol == "CUSTOM"
        assert matches[0].confidence == CONFIDENCE_PORTFOLIO

    def test_portfolio_with_solana_tokens(self, service):
        portfolio = {
            "tokens": [
                {
                    "chain_id": SOLANA_CHAIN_ID,
                    "symbol": "MYCOIN",
                    "name": "My Solana Coin",
                    "address": "MyMintAddress123456789012345678901234567890",
                    "decimals": 9,
                }
            ]
        }
        matches = service.resolve_sync("MYCOIN", portfolio=portfolio)
        assert len(matches) >= 1
        assert matches[0].chain_id == SOLANA_CHAIN_ID

    def test_portfolio_chain_filter(self, service):
        portfolio = {
            "tokens": [
                {"chain_id": 1, "symbol": "TOK", "name": "Eth Token", "address": "0x1", "decimals": 18},
                {"chain_id": 137, "symbol": "TOK", "name": "Polygon Token", "address": "0x2", "decimals": 18},
            ]
        }
        matches = service.resolve_sync("TOK", chain_id=1, portfolio=portfolio)
        # Should only return the Ethereum token
        assert all(m.chain_id == 1 for m in matches if m.source == ResolutionSource.PORTFOLIO)


# =============================================================================
# Async Resolution Tests
# =============================================================================

class TestAsyncResolution:
    """Test async resolution with mock providers."""

    @pytest.mark.asyncio
    async def test_async_resolve_from_registry(self, service):
        matches = await service.resolve("ETH", chain_id=1)
        assert len(matches) >= 1
        assert matches[0].symbol == "ETH"

    @pytest.mark.asyncio
    async def test_async_resolve_solana(self, service):
        matches = await service.resolve("BONK", chain_id=SOLANA_CHAIN_ID)
        assert len(matches) >= 1
        assert matches[0].symbol == "BONK"

    @pytest.mark.asyncio
    async def test_jupiter_provider_called_for_unknown_solana_token(
        self, mock_jupiter_provider, mock_jupiter_token
    ):
        # Setup mock to return a token
        mock_token = mock_jupiter_token(
            address="NewMint123",
            symbol="NEW",
            name="New Token",
            decimals=9,
            logo_uri="https://example.com/logo.png",
            tags=["defi"],
            coingecko_id="new-token",
        )
        mock_jupiter_provider.search_by_symbol = AsyncMock(return_value=[mock_token])

        service = TokenResolutionService(jupiter_provider=mock_jupiter_provider)
        matches = await service.resolve("NEW", chain_id=SOLANA_CHAIN_ID)

        mock_jupiter_provider.search_by_symbol.assert_called()
        assert len(matches) >= 1
        assert matches[0].symbol == "NEW"
        assert matches[0].source == ResolutionSource.JUPITER

    @pytest.mark.asyncio
    async def test_coingecko_provider_called_for_unknown_evm_token(
        self, mock_coingecko_provider
    ):
        # Setup mock to return search results
        mock_coingecko_provider.search_coins = AsyncMock(return_value=[
            {"id": "new-coin", "symbol": "NEW", "name": "New Coin", "thumb": "https://..."}
        ])

        service = TokenResolutionService(coingecko_provider=mock_coingecko_provider)
        matches = await service.resolve("NEW", chain_id=1)

        mock_coingecko_provider.search_coins.assert_called()
        assert len(matches) >= 1

    @pytest.mark.asyncio
    async def test_jupiter_mint_address_lookup(
        self, mock_jupiter_provider, mock_jupiter_token
    ):
        # Use a valid Base58 address (no 0, O, I, l characters)
        mint_addr = "NewMintAddressForTest23456789abcdefghijkm"
        mock_token = mock_jupiter_token(
            address=mint_addr,
            symbol="TESTJUP",
            name="Test Jupiter Token",
            decimals=6,
        )
        mock_jupiter_provider.get_token_by_mint = AsyncMock(return_value=mock_token)

        service = TokenResolutionService(jupiter_provider=mock_jupiter_provider)
        matches = await service.resolve(mint_addr)

        mock_jupiter_provider.get_token_by_mint.assert_called_with(mint_addr)
        assert len(matches) >= 1
        assert matches[0].symbol == "TESTJUP"


# =============================================================================
# Registry Data Tests
# =============================================================================

class TestRegistryData:
    """Test that registry data is properly structured."""

    def test_evm_registry_has_ethereum(self):
        assert 1 in TOKEN_REGISTRY
        assert "ETH" in TOKEN_REGISTRY[1]
        assert "USDC" in TOKEN_REGISTRY[1]

    def test_evm_registry_has_multiple_chains(self):
        expected_chains = {1, 8453, 42161, 10, 137}
        actual_chains = set(TOKEN_REGISTRY.keys())
        assert expected_chains.issubset(actual_chains)

    def test_solana_registry_has_top_tokens(self):
        expected_tokens = {"SOL", "USDC", "USDT", "BONK", "JUP"}
        actual_tokens = set(SOLANA_TOKEN_REGISTRY.keys())
        assert expected_tokens.issubset(actual_tokens)

    def test_solana_registry_structure(self):
        sol = SOLANA_TOKEN_REGISTRY["SOL"]
        assert sol["symbol"] == "SOL"
        assert sol["decimals"] == 9
        assert sol["is_native"] is True
        assert sol["address"] == SOLANA_NATIVE_MINT

    def test_token_aliases_exist(self):
        eth = TOKEN_REGISTRY[1]["ETH"]
        assert "aliases" in eth
        assert "eth" in eth["aliases"]

        sol = SOLANA_TOKEN_REGISTRY["SOL"]
        assert "sol" in sol["aliases"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query_returns_empty(self, service):
        matches = service.resolve_sync("")
        assert matches == []

    def test_whitespace_only_query_returns_empty(self, service):
        matches = service.resolve_sync("   ")
        assert matches == []

    def test_special_characters_handled(self, service):
        matches = service.resolve_sync("USDC.e", chain_id=42161)
        if matches:
            assert matches[0].symbol == "USDC.e"

    def test_top_k_limits_results(self, service):
        matches = service.resolve_sync("USDC", top_k=2)
        assert len(matches) <= 2

    def test_sorted_by_confidence(self, service):
        matches = service.resolve_sync("USDC")
        confidences = [m.confidence for m in matches]
        assert confidences == sorted(confidences, reverse=True)


# =============================================================================
# Service Factory Tests
# =============================================================================

class TestServiceFactory:
    """Test the get_token_resolution_service factory function."""

    def test_creates_service_without_providers(self):
        service = get_token_resolution_service()
        assert isinstance(service, TokenResolutionService)

    def test_creates_service_with_coingecko(self, mock_coingecko_provider):
        service = get_token_resolution_service(coingecko_provider=mock_coingecko_provider)
        assert service._coingecko == mock_coingecko_provider

    def test_creates_service_with_jupiter(self, mock_jupiter_provider):
        service = get_token_resolution_service(jupiter_provider=mock_jupiter_provider)
        assert service._jupiter == mock_jupiter_provider
