"""
ERC-4337 UserOperation execution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.core.execution.userop import UserOperation, UserOpGasEstimate, UserOpReceipt
from app.core.policy import FeePolicyConfig
from app.providers.bundler import BundlerProvider, get_bundler_provider
from app.providers.paymaster import PaymasterRouter, PaymasterError


class UserOpExecutionError(Exception):
    """UserOperation execution error."""
    pass


@dataclass
class UserOpExecutionResult:
    user_op_hash: str
    receipt: Optional[UserOpReceipt] = None


class UserOpExecutor:
    """
    Executes ERC-4337 UserOperations via a bundler and optional paymaster routing.
    """

    def __init__(
        self,
        bundler: Optional[BundlerProvider] = None,
        paymaster_router: Optional[PaymasterRouter] = None,
        entry_point: Optional[str] = None,
    ) -> None:
        self.bundler = bundler or get_bundler_provider()
        self.paymaster_router = paymaster_router or PaymasterRouter()
        self.entry_point = entry_point or settings.erc4337_entrypoint_address

    async def sponsor_user_operation(
        self,
        user_op: UserOperation,
        fee_config: FeePolicyConfig,
    ) -> UserOperation:
        if not self.entry_point:
            raise UserOpExecutionError("EntryPoint address is required for paymaster routing")

        try:
            paymaster_and_data = await self.paymaster_router.get_paymaster_and_data(
                user_op=user_op,
                entry_point=self.entry_point,
                fee_config=fee_config,
            )
        except PaymasterError as exc:
            raise UserOpExecutionError(str(exc)) from exc

        user_op.paymaster_and_data = paymaster_and_data
        return user_op

    async def estimate_gas(
        self,
        user_op: UserOperation,
    ) -> UserOpGasEstimate:
        if not self.entry_point:
            raise UserOpExecutionError("EntryPoint address is required for gas estimation")
        return await self.bundler.estimate_user_operation_gas(user_op, self.entry_point)

    async def send_user_operation(
        self,
        user_op: UserOperation,
    ) -> UserOpExecutionResult:
        if not self.entry_point:
            raise UserOpExecutionError("EntryPoint address is required to send user operation")

        user_op_hash = await self.bundler.send_user_operation(user_op, self.entry_point)
        return UserOpExecutionResult(user_op_hash=user_op_hash)

