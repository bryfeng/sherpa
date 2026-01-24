"""
ERC-4337 UserOperation models and helpers.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _to_hex(value: int) -> str:
    return hex(value)


@dataclass
class UserOperation:
    """
    ERC-4337 UserOperation payload.

    Values should be supplied in raw units (wei / gas units) and are encoded
    as hex for RPC calls.
    """
    sender: str
    nonce: int
    init_code: str
    call_data: str
    call_gas_limit: int
    verification_gas_limit: int
    pre_verification_gas: int
    max_fee_per_gas: int
    max_priority_fee_per_gas: int
    paymaster_and_data: str = "0x"
    signature: str = "0x"

    def to_rpc_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "nonce": _to_hex(self.nonce),
            "initCode": self.init_code,
            "callData": self.call_data,
            "callGasLimit": _to_hex(self.call_gas_limit),
            "verificationGasLimit": _to_hex(self.verification_gas_limit),
            "preVerificationGas": _to_hex(self.pre_verification_gas),
            "maxFeePerGas": _to_hex(self.max_fee_per_gas),
            "maxPriorityFeePerGas": _to_hex(self.max_priority_fee_per_gas),
            "paymasterAndData": self.paymaster_and_data,
            "signature": self.signature,
        }


@dataclass
class UserOpGasEstimate:
    call_gas_limit: int
    verification_gas_limit: int
    pre_verification_gas: int
    paymaster_verification_gas_limit: Optional[int] = None
    paymaster_post_op_gas_limit: Optional[int] = None

    @classmethod
    def from_rpc(cls, data: Dict[str, Any]) -> "UserOpGasEstimate":
        def parse_hex(value: Optional[str]) -> Optional[int]:
            if value is None:
                return None
            return int(value, 16)

        return cls(
            call_gas_limit=parse_hex(data.get("callGasLimit")) or 0,
            verification_gas_limit=parse_hex(data.get("verificationGasLimit")) or 0,
            pre_verification_gas=parse_hex(data.get("preVerificationGas")) or 0,
            paymaster_verification_gas_limit=parse_hex(data.get("paymasterVerificationGasLimit")),
            paymaster_post_op_gas_limit=parse_hex(data.get("paymasterPostOpGasLimit")),
        )


@dataclass
class UserOpReceipt:
    user_op_hash: str
    success: bool
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None

