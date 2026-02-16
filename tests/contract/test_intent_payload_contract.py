"""
Contract tests for the DCA executor → Convex smartSessionIntents:backendCreate payload.

Verifies the exact shape of the intent payload that the DCA executor sends to Convex.
Catches mismatches where tokenIn/tokenOut are strings instead of objects.

Regression triggers:
- tokenIn sent as "USDC" string instead of {symbol, address, amount} object
- address field is zero address (token not resolved)
- address missing 0x prefix
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from app.core.strategies.dca.models import (
    DCAStrategy,
    DCAConfig,
    DCAStats,
    DCAStatus,
    DCAFrequency,
    TokenInfo,
    ExecutionQuote,
)


def _make_strategy(
    from_symbol: str = "USDC",
    from_address: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    to_symbol: str = "ETH",
    to_address: str = "0x4200000000000000000000000000000000000006",
    chain_id: int = 8453,
) -> DCAStrategy:
    """Build a test DCAStrategy with specified token addresses."""
    return DCAStrategy(
        id="strat_123",
        user_id="user_1",
        wallet_id="wallet_1",
        wallet_address="0xABCDEF1234567890abcdef1234567890ABCDEF12",
        name="Test DCA",
        config=DCAConfig(
            from_token=TokenInfo(symbol=from_symbol, address=from_address, chain_id=chain_id, decimals=6),
            to_token=TokenInfo(symbol=to_symbol, address=to_address, chain_id=chain_id, decimals=18),
            amount_per_execution_usd=Decimal("5"),
            frequency=DCAFrequency.DAILY,
        ),
        status=DCAStatus.ACTIVE,
        smart_session_id="session_abc",
        stats=DCAStats(),
    )


def _make_quote() -> ExecutionQuote:
    return ExecutionQuote(
        input_amount=Decimal("5"),
        expected_output_amount=Decimal("0.002"),
        minimum_output_amount=Decimal("0.0019"),
        price_impact_bps=10,
        route=None,
        raw_quote=None,
    )


class TestIntentPayloadContract:
    """Verify the exact payload shape sent to smartSessionIntents:backendCreate."""

    @pytest.mark.asyncio
    async def test_token_in_is_object_not_string(self):
        """tokenIn must be {symbol, address, amount}, never a raw string."""
        mock_convex = AsyncMock()
        mock_convex.mutation = AsyncMock(return_value="intent_1")

        strategy = _make_strategy()
        quote = _make_quote()

        # Import and instantiate executor with mock
        from app.core.strategies.dca.executor import DCAExecutor
        executor = DCAExecutor.__new__(DCAExecutor)
        executor._convex = mock_convex

        await executor._create_ui_intent_record(strategy, quote, 8453)

        # Verify mutation was called
        assert mock_convex.mutation.called
        call_args = mock_convex.mutation.call_args
        payload = call_args[0][1]  # second positional arg is the payload dict

        # tokenIn must be a dict, not a string
        assert isinstance(payload["tokenIn"], dict), (
            f"tokenIn should be a dict, got {type(payload['tokenIn'])}: {payload['tokenIn']}"
        )
        assert "symbol" in payload["tokenIn"]
        assert "address" in payload["tokenIn"]
        assert "amount" in payload["tokenIn"]

    @pytest.mark.asyncio
    async def test_token_out_is_object_not_string(self):
        """tokenOut must be {symbol, address}, never a raw string."""
        mock_convex = AsyncMock()
        mock_convex.mutation = AsyncMock(return_value="intent_1")

        strategy = _make_strategy()
        quote = _make_quote()

        from app.core.strategies.dca.executor import DCAExecutor
        executor = DCAExecutor.__new__(DCAExecutor)
        executor._convex = mock_convex

        await executor._create_ui_intent_record(strategy, quote, 8453)

        payload = mock_convex.mutation.call_args[0][1]

        assert isinstance(payload["tokenOut"], dict), (
            f"tokenOut should be a dict, got {type(payload['tokenOut'])}: {payload['tokenOut']}"
        )
        assert "symbol" in payload["tokenOut"]
        assert "address" in payload["tokenOut"]

    @pytest.mark.asyncio
    async def test_address_is_hex_prefixed(self):
        """tokenIn.address and tokenOut.address must start with '0x'."""
        mock_convex = AsyncMock()
        mock_convex.mutation = AsyncMock(return_value="intent_1")

        strategy = _make_strategy()
        quote = _make_quote()

        from app.core.strategies.dca.executor import DCAExecutor
        executor = DCAExecutor.__new__(DCAExecutor)
        executor._convex = mock_convex

        await executor._create_ui_intent_record(strategy, quote, 8453)

        payload = mock_convex.mutation.call_args[0][1]
        assert payload["tokenIn"]["address"].startswith("0x"), (
            f"tokenIn.address missing 0x prefix: {payload['tokenIn']['address']}"
        )
        assert payload["tokenOut"]["address"].startswith("0x"), (
            f"tokenOut.address missing 0x prefix: {payload['tokenOut']['address']}"
        )

    @pytest.mark.asyncio
    async def test_address_is_not_zero(self):
        """tokenIn.address must not be the zero address (means token wasn't resolved)."""
        mock_convex = AsyncMock()
        mock_convex.mutation = AsyncMock(return_value="intent_1")

        strategy = _make_strategy()
        quote = _make_quote()

        from app.core.strategies.dca.executor import DCAExecutor
        executor = DCAExecutor.__new__(DCAExecutor)
        executor._convex = mock_convex

        await executor._create_ui_intent_record(strategy, quote, 8453)

        payload = mock_convex.mutation.call_args[0][1]
        zero_addr = "0x" + "0" * 40
        assert payload["tokenIn"]["address"] != zero_addr, (
            "tokenIn.address is zero — token was not resolved"
        )
        assert payload["tokenOut"]["address"] != zero_addr, (
            "tokenOut.address is zero — token was not resolved"
        )
