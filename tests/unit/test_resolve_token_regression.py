"""
Regression tests for _resolve_token() and DCAStrategy.from_strategies_table().

These tests guard the boundary between AI chat config (flat snake_case strings)
and DCA execution config (nested camelCase TokenInfo objects).

Regression triggers:
- "USDC" passed where address "0xA0b86991..." expected
- AI chat config format mismatch with DCA format
"""

import pytest
from decimal import Decimal
from datetime import datetime

from app.core.strategies.dca.models import (
    _resolve_token,
    DCAStrategy,
    TokenInfo,
    DCAStatus,
)


# =============================================================================
# _resolve_token() tests
# =============================================================================


class TestResolveTokenRegression:
    """Tests for _resolve_token() — the symbol-to-address resolver."""

    def test_known_symbol_usdc_base_produces_real_address(self):
        """USDC on Base → real checksum address, not zero address."""
        result = _resolve_token("USDC", chain_id=8453)
        assert isinstance(result, TokenInfo)
        assert result.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert result.symbol == "USDC"
        assert result.decimals == 6
        assert result.chain_id == 8453

    def test_known_symbol_usdc_ethereum(self):
        """USDC on Ethereum mainnet → correct address."""
        result = _resolve_token("USDC", chain_id=1)
        assert result.address == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        assert result.decimals == 6

    def test_known_symbol_eth_mainnet(self):
        """ETH on mainnet → WETH address."""
        result = _resolve_token("ETH", chain_id=1)
        assert result.address == "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        assert result.decimals == 18

    def test_known_symbol_eth_base(self):
        """ETH on Base → Base WETH address."""
        result = _resolve_token("ETH", chain_id=8453)
        assert result.address == "0x4200000000000000000000000000000000000006"
        assert result.decimals == 18

    def test_dict_with_full_info_passthrough(self):
        """Dict with all fields → TokenInfo via from_dict()."""
        result = _resolve_token({
            "symbol": "USDC",
            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "chainId": 8453,
            "decimals": 6,
        })
        assert result.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert result.symbol == "USDC"

    def test_dict_without_address_resolves_by_symbol(self):
        """Dict with symbol but no address → resolved via known tokens (Guard A)."""
        result = _resolve_token({"symbol": "USDC"}, chain_id=8453)
        assert result.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert result.symbol == "USDC"

    def test_unknown_symbol_returns_zero_address(self):
        """Unknown token symbol → zero address fallback (documents current behavior)."""
        result = _resolve_token("SHIB", chain_id=1)
        assert result.address == "0x" + "0" * 40
        assert result.symbol == "SHIB"

    def test_case_insensitive_symbol(self):
        """Lowercase 'usdc' resolves same as 'USDC'."""
        lower = _resolve_token("usdc", chain_id=8453)
        upper = _resolve_token("USDC", chain_id=8453)
        assert lower.address == upper.address
        assert lower.symbol == upper.symbol

    def test_weth_alias(self):
        """'WETH' and 'ETH' resolve to same address."""
        eth = _resolve_token("ETH", chain_id=1)
        weth = _resolve_token("WETH", chain_id=1)
        assert eth.address == weth.address

    def test_unknown_chain_returns_zero_address(self):
        """Known symbol on unknown chain → zero address."""
        result = _resolve_token("USDC", chain_id=42161)  # Arbitrum not in known tokens
        assert result.address == "0x" + "0" * 40


# =============================================================================
# DCAStrategy.from_strategies_table() tests
# =============================================================================


class TestFromStrategiesTableRegression:
    """Tests for DCAStrategy.from_strategies_table() — config format handling."""

    BASE_DOC = {
        "_id": "test_strategy_1",
        "userId": "user_1",
        "walletId": "wallet_1",
        "walletAddress": "0xABCDEF1234567890abcdef1234567890ABCDEF12",
        "name": "Test DCA",
        "status": "active",
        "totalExecutions": 0,
        "successfulExecutions": 0,
        "failedExecutions": 0,
        "createdAt": int(datetime(2025, 1, 1).timestamp() * 1000),
        "updatedAt": int(datetime(2025, 1, 1).timestamp() * 1000),
    }

    def test_ai_chat_snake_case_config(self):
        """AI chat flow creates snake_case configs with symbol strings."""
        doc = {
            **self.BASE_DOC,
            "config": {
                "from_token": "USDC",
                "to_token": "ETH",
                "amount_usd": 5,
                "frequency": "daily",
                "chain_id": 8453,
            },
        }
        strategy = DCAStrategy.from_strategies_table(doc)

        assert strategy.config.from_token.symbol == "USDC"
        assert strategy.config.from_token.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert strategy.config.to_token.symbol == "ETH"
        assert strategy.config.to_token.address == "0x4200000000000000000000000000000000000006"
        assert strategy.config.amount_per_execution_usd == Decimal("5")

    def test_canonical_camel_case_config(self):
        """DCA-specific flow creates camelCase configs with full token objects."""
        doc = {
            **self.BASE_DOC,
            "config": {
                "fromToken": {
                    "symbol": "USDC",
                    "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "chainId": 8453,
                    "decimals": 6,
                },
                "toToken": {
                    "symbol": "ETH",
                    "address": "0x4200000000000000000000000000000000000006",
                    "chainId": 8453,
                    "decimals": 18,
                },
                "amountPerExecutionUsd": 10,
                "frequency": "weekly",
                "chainId": 8453,
            },
        }
        strategy = DCAStrategy.from_strategies_table(doc)

        assert strategy.config.from_token.symbol == "USDC"
        assert strategy.config.from_token.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert strategy.config.amount_per_execution_usd == Decimal("10")

    def test_mixed_format_camel_keys_string_values(self):
        """camelCase keys with symbol strings (another common variant)."""
        doc = {
            **self.BASE_DOC,
            "config": {
                "fromToken": "USDC",
                "toToken": "ETH",
                "amount_usd": 5,
                "frequency": "daily",
                "chain_id": 8453,
            },
        }
        strategy = DCAStrategy.from_strategies_table(doc)

        assert strategy.config.from_token.symbol == "USDC"
        assert strategy.config.from_token.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert strategy.config.to_token.symbol == "ETH"

    def test_from_token_address_is_never_zero_for_known_tokens(self):
        """Regression: resolved tokens must have real addresses, never zero."""
        doc = {
            **self.BASE_DOC,
            "config": {
                "from_token": "USDC",
                "to_token": "ETH",
                "amount_usd": 5,
                "frequency": "daily",
                "chain_id": 8453,
            },
        }
        strategy = DCAStrategy.from_strategies_table(doc)

        zero_addr = "0x" + "0" * 40
        assert strategy.config.from_token.address != zero_addr, (
            f"from_token resolved to zero address for known symbol USDC"
        )
        assert strategy.config.to_token.address != zero_addr, (
            f"to_token resolved to zero address for known symbol ETH"
        )
