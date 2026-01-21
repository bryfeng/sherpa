"""
Tests for execution pre-flight hardening.
"""

import pytest
from unittest.mock import AsyncMock

from app.core.execution.executor import (
    TransactionExecutor,
    ExecutionError,
    TransactionRevertError,
    SignatureRequiredError,
)
from app.core.execution.models import (
    PreparedTransaction,
    TransactionType,
    GasEstimate,
    ExecutionContext,
    SwapQuote,
    TransactionStatus,
)


class DummyNonceManager:
    """Minimal nonce manager stub for tests."""

    def __init__(self):
        self.release_nonce = AsyncMock()
        self.confirm_nonce = AsyncMock()
        self.get_next_nonce = AsyncMock(return_value=1)


@pytest.mark.asyncio
async def test_simulation_revert_marks_failed():
    executor = TransactionExecutor()
    executor.nonce_manager = DummyNonceManager()

    tx = PreparedTransaction(
        tx_id="tx_test",
        tx_type=TransactionType.SWAP,
        chain_id=1,
        from_address="0x1111111111111111111111111111111111111111",
        to_address="0x2222222222222222222222222222222222222222",
        data="0x",
        value=0,
        gas_estimate=GasEstimate(gas_limit=21000, gas_price_wei=1, estimated_cost_wei=21000),
        nonce=1,
    )
    context = ExecutionContext(wallet_address=tx.from_address, chain_id=1, simulate=True)

    executor._simulate_transaction = AsyncMock(
        side_effect=TransactionRevertError("Simulation reverted")
    )

    result = await executor._execute_transaction(tx=tx, context=context, signed_tx=None)

    assert result.status == TransactionStatus.FAILED
    assert result.error is not None
    executor.nonce_manager.release_nonce.assert_awaited_once()


def _make_swap_quote(signatures=None, approvals=None) -> SwapQuote:
    return SwapQuote(
        request_id="req_1",
        chain_id=1,
        wallet_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        token_in_address="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        token_in_symbol="AAA",
        token_in_decimals=18,
        amount_in=100,
        token_out_address="0xcccccccccccccccccccccccccccccccccccccccc",
        token_out_symbol="BBB",
        token_out_decimals=18,
        amount_out_estimate=90,
        price_in_usd=1.0,
        price_out_usd=1.0,
        value_in_usd=100.0,
        value_out_usd=90.0,
        gas_fee_usd=1.0,
        relay_fee_usd=0.1,
        total_fee_usd=1.1,
        slippage_bps=50,
        tx={"to": "0xdddddddddddddddddddddddddddddddddddddddd", "data": "0x", "value": "0x0"},
        approvals=approvals or [],
        signatures=signatures or [],
    )


@pytest.mark.asyncio
async def test_allowance_insufficient_blocks_swap():
    executor = TransactionExecutor()
    executor._enforce_policy = AsyncMock(return_value=None)
    executor._get_allowance = AsyncMock(return_value=0)

    quote = _make_swap_quote(
        approvals=[{"data": {"spender": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "amount": "0x10"}}]
    )
    context = ExecutionContext(wallet_address=quote.wallet_address, chain_id=1)

    with pytest.raises(ExecutionError):
        await executor.execute_swap(quote=quote, context=context)


@pytest.mark.asyncio
async def test_signature_required_blocks_swap():
    executor = TransactionExecutor()
    executor._enforce_policy = AsyncMock(return_value=None)

    quote = _make_swap_quote(signatures=[{"type": "eip712"}])
    context = ExecutionContext(wallet_address=quote.wallet_address, chain_id=1)

    with pytest.raises(SignatureRequiredError):
        await executor.execute_swap(quote=quote, context=context)
