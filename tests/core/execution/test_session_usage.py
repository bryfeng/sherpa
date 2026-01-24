"""
Tests for session key usage recording in execution layer.
"""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

from app.core.execution.executor import TransactionExecutor
from app.core.execution.models import ExecutionContext
from app.core.wallet.models import Permission


@pytest.mark.asyncio
async def test_record_session_usage_calls_manager():
    manager = AsyncMock()
    executor = TransactionExecutor(session_manager=manager)

    context = ExecutionContext(
        wallet_address="0x1111111111111111111111111111111111111111",
        chain_id=1,
        session_key_id="sess_test",
        require_policy=False,
        require_session_key=False,
        simulate=True,
    )

    await executor._record_session_usage(
        context=context,
        action=Permission.SWAP,
        value_usd=100.5,
        tx_hash="0xabc",
        metadata={"note": "test"},
    )

    manager.record_usage.assert_awaited_once()
    args, kwargs = manager.record_usage.call_args
    assert kwargs["session_id"] == "sess_test"
    assert kwargs["action_type"] == Permission.SWAP
    assert kwargs["value_usd"] == Decimal("100.5")
