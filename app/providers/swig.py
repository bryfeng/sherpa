"""
Swig Smart Wallet Provider (Solana).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from .base import Provider
from ..config import settings


class SwigError(Exception):
    """Swig provider error."""
    pass


@dataclass
class SwigConfig:
    base_url: str
    api_key: str


@dataclass
class SwigSessionAuthority:
    authority_id: str
    wallet_address: str
    expires_at: Optional[datetime] = None
    status: Optional[str] = None
    raw: Dict[str, Any] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "SwigSessionAuthority":
        expires_at = None
        if data.get("expiresAt"):
            try:
                expires_at = datetime.fromisoformat(data["expiresAt"])
            except ValueError:
                expires_at = None
        return cls(
            authority_id=data.get("authorityId") or data.get("id") or "",
            wallet_address=data.get("walletAddress") or data.get("wallet") or "",
            expires_at=expires_at,
            status=data.get("status"),
            raw=data,
        )


class SwigProvider(Provider):
    name = "swig"
    timeout_s = 20

    def __init__(self, config: Optional[SwigConfig] = None) -> None:
        self._config = config or SwigConfig(
            base_url=settings.swig_base_url,
            api_key=settings.swig_api_key,
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def ready(self) -> bool:
        return bool(self._config.base_url and self._config.api_key) and settings.enable_swig

    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {"status": "disabled", "reason": "Swig not configured"}

        try:
            result = await self._request("GET", "/health")
            return {"status": "healthy", "result": result}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    async def create_session_authority(
        self,
        wallet_address: str,
        permissions: Dict[str, Any],
        expires_at: Optional[datetime] = None,
    ) -> SwigSessionAuthority:
        payload: Dict[str, Any] = {
            "walletAddress": wallet_address,
            "permissions": permissions,
        }
        if expires_at:
            payload["expiresAt"] = expires_at.isoformat()

        result = await self._request("POST", "/session-authorities", json=payload)
        if not isinstance(result, dict):
            raise SwigError("Invalid Swig response for session authority creation")
        return SwigSessionAuthority.from_api(result)

    async def revoke_session_authority(self, authority_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/session-authorities/{authority_id}/revoke", json={})

    async def build_reimbursement(self, authority_id: str, tx_signature: str) -> Dict[str, Any]:
        payload = {"txSignature": tx_signature}
        return await self._request(
            "POST",
            f"/session-authorities/{authority_id}/reimburse",
            json=payload,
        )

    async def _request(self, method: str, path: str, json: Optional[Dict[str, Any]] = None) -> Any:
        if not await self.ready():
            raise SwigError("Swig provider is not configured")

        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)

        url = f"{self._config.base_url.rstrip('/')}{path}"
        response = await self._client.request(
            method,
            url,
            json=json,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
        )
        response.raise_for_status()
        return response.json()


_swig_provider: Optional[SwigProvider] = None


def get_swig_provider() -> SwigProvider:
    global _swig_provider
    if _swig_provider is None:
        _swig_provider = SwigProvider()
    return _swig_provider

