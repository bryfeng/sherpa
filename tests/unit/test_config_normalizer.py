"""
Unit tests for strategy config normalizer.

Verifies that all config format variants are mapped to the canonical
camelCase nested format expected by the executor and frontend.
"""

import pytest

from app.core.strategies.config_normalizer import normalize_strategy_config


class TestNormalizeDCAConfig:
    """Tests for DCA strategy config normalization."""

    def test_snake_case_flat_to_canonical(self):
        """Agent-generated snake_case flat config → canonical camelCase nested."""
        raw = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "frequency": "hourly",
            "chain_id": 8453,
            "max_slippage_bps": 50,
            "max_gas_usd": 0.5,
        }

        result = normalize_strategy_config("dca", raw)

        assert result["fromToken"] == {"symbol": "USDC", "chainId": 8453}
        assert result["toToken"] == {"symbol": "ETH", "chainId": 8453}
        assert result["amountPerExecution"] == 5
        assert result["chainId"] == 8453
        assert result["maxSlippageBps"] == 50
        assert result["maxGasUsd"] == 0.5
        assert result["frequency"] == "hourly"

    def test_camel_case_nested_unchanged(self):
        """Already-canonical config passes through unchanged (idempotent)."""
        raw = {
            "fromToken": {"symbol": "USDC", "address": "0xabc", "chainId": 8453},
            "toToken": {"symbol": "ETH", "address": "0xdef", "chainId": 8453},
            "amountPerExecution": 10,
            "chainId": 8453,
            "maxSlippageBps": 100,
            "maxGasUsd": 1.0,
            "frequency": "daily",
        }

        result = normalize_strategy_config("dca", raw)

        assert result["fromToken"] == {"symbol": "USDC", "address": "0xabc", "chainId": 8453}
        assert result["toToken"] == {"symbol": "ETH", "address": "0xdef", "chainId": 8453}
        assert result["amountPerExecution"] == 10
        assert result["chainId"] == 8453
        assert result["maxSlippageBps"] == 100
        assert result["maxGasUsd"] == 1.0

    def test_idempotent_double_normalize(self):
        """Normalizing an already-normalized config produces identical output."""
        raw = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "chain_id": 8453,
        }

        first = normalize_strategy_config("dca", raw)
        second = normalize_strategy_config("dca", first)

        assert first == second

    def test_mixed_format(self):
        """Mixed snake_case and camelCase fields are handled correctly."""
        raw = {
            "fromToken": {"symbol": "USDC"},
            "to_token": "ETH",
            "amount_usd": 5,
            "chainId": 8453,
        }

        result = normalize_strategy_config("dca", raw)

        assert result["fromToken"] == {"symbol": "USDC"}
        assert result["toToken"] == {"symbol": "ETH", "chainId": 8453}
        assert result["amountPerExecution"] == 5
        assert result["chainId"] == 8453

    def test_snake_case_with_addresses(self):
        """snake_case config with explicit token addresses."""
        raw = {
            "from_token": "USDC",
            "from_token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "to_token": "ETH",
            "to_token_address": "0x4200000000000000000000000000000000000006",
            "amount_usd": 10,
            "chain_id": 8453,
        }

        result = normalize_strategy_config("dca", raw)

        assert result["fromToken"]["symbol"] == "USDC"
        assert result["fromToken"]["address"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert result["toToken"]["symbol"] == "ETH"
        assert result["toToken"]["address"] == "0x4200000000000000000000000000000000000006"

    def test_missing_fields_no_crash(self):
        """Partial config doesn't crash — missing fields are simply absent."""
        raw = {"from_token": "USDC"}

        result = normalize_strategy_config("dca", raw)

        assert result["fromToken"] == {"symbol": "USDC"}
        assert "toToken" not in result
        assert "amountPerExecution" not in result

    def test_empty_config(self):
        """Empty config returns empty dict."""
        result = normalize_strategy_config("dca", {})
        assert result == {}

    def test_none_config(self):
        """None config returns None."""
        result = normalize_strategy_config("dca", None)
        assert result is None

    def test_preserves_extra_keys(self):
        """Unknown keys are preserved in the output."""
        raw = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "customField": "hello",
            "anotherThing": 42,
        }

        result = normalize_strategy_config("dca", raw)

        assert result["customField"] == "hello"
        assert result["anotherThing"] == 42

    def test_address_as_token_string(self):
        """A 0x address passed as from_token is treated as address, not symbol."""
        raw = {
            "from_token": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "to_token": "ETH",
            "amount_usd": 5,
        }

        result = normalize_strategy_config("dca", raw)

        assert "address" in result["fromToken"]
        assert "symbol" not in result["fromToken"]

    def test_amount_per_execution_snake(self):
        """amount_per_execution (snake_case) maps to amountPerExecution."""
        raw = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_per_execution": 25,
        }

        result = normalize_strategy_config("dca", raw)

        assert result["amountPerExecution"] == 25

    def test_non_dca_passthrough(self):
        """Non-DCA strategy types pass through unchanged for now."""
        raw = {"foo": "bar", "baz": 123}

        result = normalize_strategy_config("rebalance", raw)

        assert result == raw
