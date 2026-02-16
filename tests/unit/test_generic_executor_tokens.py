"""
Regression tests for GenericStrategyExecutor._extract_swap_params().

Documents the behavior boundary where swap params are extracted from config.
The generic executor normalizes config then extracts tokens — but if the
normalizer returns a dict without an "address" key, the executor gets a
symbol string instead of an address.

Regression triggers:
- tokenIn/tokenOut sent as raw "USDC" string instead of {symbol, address, amount} object
"""

import pytest
from unittest.mock import AsyncMock

from app.core.strategies.generic_executor import GenericStrategyExecutor, SwapParams


WALLET = "0x1234567890abcdef1234567890abcdef12345678"


class TestExtractSwapParamsTokenResolution:
    """Tests for _extract_swap_params token handling."""

    def _make_executor(self) -> GenericStrategyExecutor:
        return GenericStrategyExecutor(convex_client=AsyncMock())

    def test_snake_case_extracts_symbol_not_address(self):
        """Documents: snake_case config → from_token is symbol string, NOT address.

        This is the root cause of the "USDC passed where address expected" bug.
        The normalizer converts from_token: "USDC" → fromToken: {symbol: "USDC"}
        but there's no address key, so _extract_swap_params falls through to symbol.
        """
        config = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "chain_id": 8453,
        }
        executor = self._make_executor()
        result = executor._extract_swap_params(config, WALLET, "dca")

        assert result is not None
        # Documents current behavior: from_token is the SYMBOL, not an address
        assert result.from_token == "USDC"
        assert not result.from_token.startswith("0x"), (
            "Expected symbol string, got address — behavior has changed"
        )

    def test_camel_case_with_address_extracts_address(self):
        """camelCase config with address → from_token IS the address."""
        config = {
            "fromToken": {
                "symbol": "USDC",
                "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "chainId": 8453,
            },
            "toToken": {
                "symbol": "ETH",
                "address": "0x4200000000000000000000000000000000000006",
                "chainId": 8453,
            },
            "amountPerExecution": 10,
            "chainId": 8453,
        }
        executor = self._make_executor()
        result = executor._extract_swap_params(config, WALLET, "dca")

        assert result is not None
        assert result.from_token == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert result.to_token == "0x4200000000000000000000000000000000000006"

    def test_symbol_only_nested_dict_falls_back_to_symbol(self):
        """Nested dict with symbol but no address → falls back to symbol."""
        config = {
            "fromToken": {"symbol": "USDC", "chainId": 8453},
            "toToken": {"symbol": "ETH", "chainId": 8453},
            "amountPerExecution": 5,
            "chainId": 8453,
        }
        executor = self._make_executor()
        result = executor._extract_swap_params(config, WALLET, "dca")

        assert result is not None
        # No address in dict → falls through to symbol
        assert result.from_token == "USDC"
        assert result.to_token == "ETH"

    def test_amount_extraction_from_snake_case(self):
        """amount_usd in snake_case config → properly extracted."""
        config = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 25,
            "chain_id": 8453,
        }
        executor = self._make_executor()
        result = executor._extract_swap_params(config, WALLET, "dca")

        assert result is not None
        assert result.amount == "25"

    def test_missing_tokens_returns_none(self):
        """Config with no token info at all → returns None."""
        config = {"amount": 5, "chain_id": 8453}
        executor = self._make_executor()
        result = executor._extract_swap_params(config, WALLET, "dca")

        assert result is None
