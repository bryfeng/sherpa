"""
ERC-4337 Paymaster Provider and routing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .base import Provider
from ..config import settings
from ..core.execution.userop import UserOperation
from ..core.policy import FeePolicyConfig


class PaymasterError(Exception):
    """Paymaster provider error."""
    pass


@dataclass
class PaymasterConfig:
    rpc_url: str
    rpc_method: str = "pm_sponsorUserOperation"


class PaymasterProvider(Provider):
    name = "paymaster"
    timeout_s = 20

    def __init__(self, config: Optional[PaymasterConfig] = None) -> None:
        self._config = config or PaymasterConfig(
            rpc_url=settings.erc4337_paymaster_url,
            rpc_method=settings.erc4337_paymaster_rpc_method,
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def ready(self) -> bool:
        return bool(self._config.rpc_url) and settings.enable_erc4337

    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {"status": "disabled", "reason": "Paymaster not configured"}

        try:
            result = await self._rpc_call("eth_chainId", [])
            return {"status": "healthy", "chainId": result}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    async def sponsor_user_operation(
        self,
        user_op: UserOperation,
        entry_point: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not await self.ready():
            raise PaymasterError("Paymaster provider is not configured")

        params = [user_op.to_rpc_dict(), entry_point]
        if context:
            params.append(context)
        result = await self._rpc_call(self._config.rpc_method, params)
        if isinstance(result, dict):
            paymaster_and_data = result.get("paymasterAndData") or result.get("paymaster_and_data")
            if paymaster_and_data:
                return paymaster_and_data
        if isinstance(result, str):
            return result
        raise PaymasterError("Invalid paymaster response")

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
            raise PaymasterError(payload["error"])
        return payload.get("result")


class PaymasterRouter:
    def __init__(self, paymaster: Optional[PaymasterProvider] = None) -> None:
        self.paymaster = paymaster or get_paymaster_provider()

    async def get_paymaster_and_data(
        self,
        user_op: UserOperation,
        entry_point: str,
        fee_config: FeePolicyConfig,
    ) -> str:
        if not await self.paymaster.ready():
            raise PaymasterError("Paymaster provider is not ready")

        if fee_config.stablecoin_symbol.upper() != "USDC":
            raise PaymasterError("Paymaster routing expects USDC-first fee policy")

        context = {
            "feeAssetOrder": fee_config.fee_asset_order,
            "stablecoinAddress": fee_config.stablecoin_address,
            "stablecoinSymbol": fee_config.stablecoin_symbol,
            "nativeSymbol": fee_config.native_symbol,
            "allowNativeFallback": fee_config.allow_native_fallback,
        }
        return await self.paymaster.sponsor_user_operation(
            user_op=user_op,
            entry_point=entry_point,
            context=context,
        )


_paymaster_provider: Optional[PaymasterProvider] = None


def get_paymaster_provider() -> PaymasterProvider:
    global _paymaster_provider
    if _paymaster_provider is None:
        _paymaster_provider = PaymasterProvider()
    return _paymaster_provider

