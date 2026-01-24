"""
ERC-4337 Bundler Provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .base import Provider
from ..config import settings
from ..core.execution.userop import UserOperation, UserOpGasEstimate, UserOpReceipt


class BundlerError(Exception):
    """Bundler provider error."""
    pass


@dataclass
class BundlerConfig:
    rpc_url: str


class BundlerProvider(Provider):
    name = "bundler"
    timeout_s = 20

    def __init__(self, config: Optional[BundlerConfig] = None) -> None:
        self._config = config or BundlerConfig(rpc_url=settings.erc4337_bundler_url)
        self._client: Optional[httpx.AsyncClient] = None

    async def ready(self) -> bool:
        return bool(self._config.rpc_url) and settings.enable_erc4337

    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {"status": "disabled", "reason": "Bundler not configured"}

        try:
            result = await self._rpc_call("eth_chainId", [])
            return {"status": "healthy", "chainId": result}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    async def send_user_operation(
        self,
        user_op: UserOperation,
        entry_point: str,
    ) -> str:
        if not await self.ready():
            raise BundlerError("Bundler provider is not configured")

        result = await self._rpc_call(
            "eth_sendUserOperation",
            [user_op.to_rpc_dict(), entry_point],
        )
        if not isinstance(result, str):
            raise BundlerError("Invalid bundler response for eth_sendUserOperation")
        return result

    async def estimate_user_operation_gas(
        self,
        user_op: UserOperation,
        entry_point: str,
    ) -> UserOpGasEstimate:
        if not await self.ready():
            raise BundlerError("Bundler provider is not configured")

        result = await self._rpc_call(
            "eth_estimateUserOperationGas",
            [user_op.to_rpc_dict(), entry_point],
        )
        if not isinstance(result, dict):
            raise BundlerError("Invalid bundler response for eth_estimateUserOperationGas")
        return UserOpGasEstimate.from_rpc(result)

    async def get_user_operation_receipt(self, user_op_hash: str) -> Optional[UserOpReceipt]:
        if not await self.ready():
            raise BundlerError("Bundler provider is not configured")

        result = await self._rpc_call(
            "eth_getUserOperationReceipt",
            [user_op_hash],
        )
        if not result:
            return None

        receipt = result.get("receipt") or {}
        return UserOpReceipt(
            user_op_hash=user_op_hash,
            success=receipt.get("status") == "0x1",
            transaction_hash=receipt.get("transactionHash"),
            block_number=int(receipt.get("blockNumber"), 16) if receipt.get("blockNumber") else None,
            gas_used=int(receipt.get("gasUsed"), 16) if receipt.get("gasUsed") else None,
        )

    async def _rpc_call(self, method: str, params: list[Any]) -> Any:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)

        response = await self._client.post(
            self._config.rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise BundlerError(payload["error"])
        return payload.get("result")


_bundler_provider: Optional[BundlerProvider] = None


def get_bundler_provider() -> BundlerProvider:
    global _bundler_provider
    if _bundler_provider is None:
        _bundler_provider = BundlerProvider()
    return _bundler_provider

