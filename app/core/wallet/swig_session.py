"""
Swig session authority manager for Solana smart wallets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional

from app.providers.swig import SwigProvider, SwigSessionAuthority, get_swig_provider


class SwigSessionError(Exception):
    """Swig session authority error."""
    pass


@dataclass
class SwigSessionConfig:
    wallet_address: str
    max_value_usd: Decimal
    allowed_programs: list[str]
    expires_in_hours: int = 24
    metadata: Optional[Dict[str, Any]] = None


class SwigSessionAuthorityManager:
    """
    Creates and manages Swig session authorities for Solana smart wallets.
    """

    def __init__(self, provider: Optional[SwigProvider] = None) -> None:
        self._provider = provider or get_swig_provider()

    async def create_authority(self, config: SwigSessionConfig) -> SwigSessionAuthority:
        expires_at = datetime.utcnow() + timedelta(hours=config.expires_in_hours)
        permissions = {
            "maxValueUsd": str(config.max_value_usd),
            "allowedPrograms": config.allowed_programs,
            "metadata": config.metadata or {},
        }
        return await self._provider.create_session_authority(
            wallet_address=config.wallet_address,
            permissions=permissions,
            expires_at=expires_at,
        )

    async def revoke_authority(self, authority_id: str) -> Dict[str, Any]:
        return await self._provider.revoke_session_authority(authority_id)

    async def reimburse(self, authority_id: str, tx_signature: str) -> Dict[str, Any]:
        return await self._provider.build_reimbursement(authority_id, tx_signature)

