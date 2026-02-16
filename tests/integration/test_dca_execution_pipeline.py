"""
Integration test for the full DCA execution pipeline.

Tests the golden path: AI chat config → token resolution → intent payload.

Regression triggers:
- AI chat config (flat snake_case) not properly converted to DCA format
- Token symbols not resolved to addresses before Convex mutation
- Intent payload shape wrong for Convex
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.core.strategies.dca.models import (
    DCAStrategy,
    DCAConfig,
    DCAStats,
    DCAStatus,
    DCAFrequency,
    TokenInfo,
    ExecutionQuote,
)


class TestDCAExecutionPipelineGoldenPath:
    """Full pipeline: AI chat config → token resolution → intent payload."""

    AI_CHAT_STRATEGY_DOC = {
        "_id": "strat_pipeline_test",
        "userId": "user_1",
        "walletId": "wallet_1",
        "walletAddress": "0xABCDEF1234567890abcdef1234567890ABCDEF12",
        "name": "Buy ETH weekly",
        "status": "active",
        "smartSessionId": "session_abc",
        "totalExecutions": 0,
        "successfulExecutions": 0,
        "failedExecutions": 0,
        "createdAt": int(datetime(2025, 1, 1).timestamp() * 1000),
        "updatedAt": int(datetime(2025, 1, 1).timestamp() * 1000),
        "config": {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "frequency": "daily",
            "chain_id": 8453,
        },
    }

    def test_step1_from_strategies_table_resolves_tokens(self):
        """Step 1: from_strategies_table() resolves symbol strings to real addresses."""
        strategy = DCAStrategy.from_strategies_table(self.AI_CHAT_STRATEGY_DOC)

        # Tokens must be fully resolved
        assert strategy.config.from_token.symbol == "USDC"
        assert strategy.config.from_token.address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert strategy.config.from_token.chain_id == 8453
        assert strategy.config.from_token.decimals == 6

        assert strategy.config.to_token.symbol == "ETH"
        assert strategy.config.to_token.address == "0x4200000000000000000000000000000000000006"
        assert strategy.config.to_token.chain_id == 8453
        assert strategy.config.to_token.decimals == 18

    def test_step2_addresses_are_never_zero(self):
        """Step 2: No token address is the zero address after resolution."""
        strategy = DCAStrategy.from_strategies_table(self.AI_CHAT_STRATEGY_DOC)
        zero_addr = "0x" + "0" * 40

        assert strategy.config.from_token.address != zero_addr
        assert strategy.config.to_token.address != zero_addr

    @pytest.mark.asyncio
    async def test_step3_intent_payload_has_correct_shape(self):
        """Step 3: _create_ui_intent_record() sends properly shaped payload to Convex."""
        strategy = DCAStrategy.from_strategies_table(self.AI_CHAT_STRATEGY_DOC)

        quote = ExecutionQuote(
            input_amount=Decimal("5"),
            expected_output_amount=Decimal("0.002"),
            minimum_output_amount=Decimal("0.0019"),
            price_impact_bps=10,
            route=None,
            raw_quote=None,
        )

        mock_convex = AsyncMock()
        mock_convex.mutation = AsyncMock(return_value="intent_123")

        from app.core.strategies.dca.executor import DCAExecutor
        executor = DCAExecutor.__new__(DCAExecutor)
        executor._convex = mock_convex

        await executor._create_ui_intent_record(strategy, quote, 8453)

        # Verify mutation called with correct shape
        assert mock_convex.mutation.called
        call_args = mock_convex.mutation.call_args
        mutation_name = call_args[0][0]
        payload = call_args[0][1]

        assert mutation_name == "smartSessionIntents:backendCreate"

        # tokenIn is a dict with real USDC address
        assert isinstance(payload["tokenIn"], dict)
        assert payload["tokenIn"]["symbol"] == "USDC"
        assert payload["tokenIn"]["address"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert payload["tokenIn"]["amount"] == "5"

        # tokenOut is a dict with real ETH/WETH address
        assert isinstance(payload["tokenOut"], dict)
        assert payload["tokenOut"]["symbol"] == "ETH"
        assert payload["tokenOut"]["address"] == "0x4200000000000000000000000000000000000006"

        # Other required fields
        assert payload["smartSessionId"] == "session_abc"
        assert payload["chainId"] == 8453
        assert payload["intentType"] == "dca_execution"
        assert payload["sourceType"] == "dca_strategy"
        assert payload["sourceId"] == "strat_pipeline_test"

    @pytest.mark.asyncio
    async def test_full_pipeline_golden_path(self):
        """End-to-end: AI chat doc → strategy → intent payload with real addresses."""
        # 1. Parse strategy from AI chat format
        strategy = DCAStrategy.from_strategies_table(self.AI_CHAT_STRATEGY_DOC)

        # 2. Verify token resolution
        assert strategy.config.from_token.address.startswith("0x")
        assert strategy.config.to_token.address.startswith("0x")
        assert strategy.config.from_token.address != "0x" + "0" * 40
        assert strategy.config.to_token.address != "0x" + "0" * 40

        # 3. Build intent payload (as _create_ui_intent_record does)
        config = strategy.config
        quote = ExecutionQuote(
            input_amount=Decimal("5"),
            expected_output_amount=Decimal("0.002"),
            minimum_output_amount=Decimal("0.0019"),
            price_impact_bps=10,
            route=None,
            raw_quote=None,
        )

        payload = {
            "smartSessionId": strategy.smart_session_id,
            "smartAccountAddress": strategy.wallet_address,
            "intentType": "dca_execution",
            "sourceType": "dca_strategy",
            "sourceId": strategy.id,
            "chainId": 8453,
            "estimatedValueUsd": float(quote.input_amount),
            "tokenIn": {
                "symbol": config.from_token.symbol,
                "address": config.from_token.address,
                "amount": str(quote.input_amount),
            },
            "tokenOut": {
                "symbol": config.to_token.symbol,
                "address": config.to_token.address,
                "amount": str(quote.expected_output_amount),
            },
        }

        # 4. Validate every field
        assert payload["tokenIn"] == {
            "symbol": "USDC",
            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "amount": "5",
        }
        assert payload["tokenOut"] == {
            "symbol": "ETH",
            "address": "0x4200000000000000000000000000000000000006",
            "amount": "0.002",
        }
        assert payload["smartSessionId"] == "session_abc"
        assert payload["chainId"] == 8453
