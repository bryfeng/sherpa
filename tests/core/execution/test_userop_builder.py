"""
Tests for ERC-4337 UserOperation calldata builders.
"""

from app.core.execution.userop_builder import build_execute_call_data, get_execute_selector


def test_build_execute_call_data_encodes_execute() -> None:
    selector = get_execute_selector("execute(address,uint256,bytes)", None)
    call_data = build_execute_call_data(
        to_address="0x1111111111111111111111111111111111111111",
        value_wei=1,
        data="0x1234",
        signature="execute(address,uint256,bytes)",
    )

    assert call_data.startswith(selector)
    # 4-byte selector + 3 words (address, value, offset) + bytes length + data padded
    assert len(call_data) == len(selector) + 64 * 4 + 64
    assert call_data.endswith("1234" + "0" * 60)
