"""
Session wallet provisioning service.

Handles creation and management of Turnkey session wallets for autonomous execution.
These wallets hold the signing keys that are registered as session keys on user SCWs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.db import get_convex_client
from app.providers.turnkey import (
    get_turnkey_provider,
    TurnkeyError,
    TurnkeyWallet,
)

logger = logging.getLogger(__name__)


class SessionWalletError(Exception):
    """Session wallet service error."""
    pass


@dataclass
class SessionWalletInfo:
    """Session wallet information."""
    wallet_address: str  # User's main wallet
    chain_type: str  # "evm" or "solana"
    turnkey_wallet_id: str
    turnkey_address: str  # The address to register as session key
    status: str
    created_at: int
    total_signatures: int = 0
    last_used_at: Optional[int] = None


class SessionWalletService:
    """
    Manages Turnkey session wallets for users.

    This service:
    1. Creates Turnkey wallets for users (one per chain type)
    2. Stores wallet references in Convex
    3. Provides signing addresses to register as session keys on SCWs
    4. Tracks signature usage

    Usage:
        service = SessionWalletService()

        # Get or create session wallet for a user
        info = await service.get_or_create_session_wallet(
            wallet_address="0x123...",
            chain_type="evm",
        )

        # The turnkey_address should be granted session key permissions
        # on the user's smart contract wallet
        print(f"Register {info.turnkey_address} as session key on your SCW")

        # Later, get the signing address for execution
        address = await service.get_signing_address("0x123...", "evm")
    """

    def __init__(self) -> None:
        self._convex = get_convex_client()
        self._turnkey = get_turnkey_provider()

    async def get_or_create_session_wallet(
        self,
        wallet_address: str,
        chain_type: str = "evm",
        label: Optional[str] = None,
    ) -> SessionWalletInfo:
        """
        Get existing session wallet or create a new one.

        Args:
            wallet_address: User's main wallet address
            chain_type: "evm" for Ethereum/L2s, "solana" for Solana
            label: Optional label for the wallet

        Returns:
            SessionWalletInfo with Turnkey wallet details

        Raises:
            SessionWalletError: If creation fails
        """
        wallet_address = wallet_address.lower()

        # Check if exists in Convex
        existing = await self._convex.query(
            "sessionWallets:getByWalletChain",
            {"walletAddress": wallet_address, "chainType": chain_type},
        )

        if existing and existing.get("status") == "active":
            logger.info(
                f"Found existing session wallet for {wallet_address} ({chain_type})"
            )
            return SessionWalletInfo(
                wallet_address=wallet_address,
                chain_type=chain_type,
                turnkey_wallet_id=existing["turnkeyWalletId"],
                turnkey_address=existing["turnkeyAddress"],
                status=existing["status"],
                created_at=existing["createdAt"],
                total_signatures=existing.get("totalSignatures", 0),
                last_used_at=existing.get("lastUsedAt"),
            )

        # Create new Turnkey wallet
        logger.info(f"Creating new Turnkey wallet for {wallet_address} ({chain_type})")

        try:
            turnkey_wallet = await self._turnkey.create_session_wallet(
                user_id=wallet_address,
                chain_type=chain_type,
            )
        except TurnkeyError as e:
            logger.error(f"Failed to create Turnkey wallet: {e}")
            raise SessionWalletError(f"Failed to create session wallet: {e}") from e

        # Store in Convex
        try:
            await self._convex.mutation(
                "sessionWallets:create",
                {
                    "walletAddress": wallet_address,
                    "chainType": chain_type,
                    "turnkeyWalletId": turnkey_wallet.wallet_id,
                    "turnkeyAddress": turnkey_wallet.address,
                    "label": label,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store session wallet in Convex: {e}")
            # Don't raise - the Turnkey wallet exists, we can retry storing later
            logger.warning("Session wallet created in Turnkey but not stored in Convex")

        logger.info(
            f"Created session wallet {turnkey_wallet.wallet_id} "
            f"with address {turnkey_wallet.address}"
        )

        return SessionWalletInfo(
            wallet_address=wallet_address,
            chain_type=chain_type,
            turnkey_wallet_id=turnkey_wallet.wallet_id,
            turnkey_address=turnkey_wallet.address,
            status="active",
            created_at=int(time.time() * 1000),
            total_signatures=0,
        )

    async def get_signing_address(
        self,
        wallet_address: str,
        chain_type: str = "evm",
    ) -> Optional[str]:
        """
        Get the Turnkey signing address for a user.

        Args:
            wallet_address: User's main wallet address
            chain_type: "evm" or "solana"

        Returns:
            Turnkey address if exists, None otherwise
        """
        wallet_address = wallet_address.lower()

        existing = await self._convex.query(
            "sessionWallets:getByWalletChain",
            {"walletAddress": wallet_address, "chainType": chain_type},
        )

        if existing and existing.get("status") == "active":
            return existing.get("turnkeyAddress")

        return None

    async def get_session_wallet(
        self,
        wallet_address: str,
        chain_type: str = "evm",
    ) -> Optional[SessionWalletInfo]:
        """
        Get session wallet info if it exists.

        Args:
            wallet_address: User's main wallet address
            chain_type: "evm" or "solana"

        Returns:
            SessionWalletInfo if exists, None otherwise
        """
        wallet_address = wallet_address.lower()

        existing = await self._convex.query(
            "sessionWallets:getByWalletChain",
            {"walletAddress": wallet_address, "chainType": chain_type},
        )

        if not existing:
            return None

        return SessionWalletInfo(
            wallet_address=wallet_address,
            chain_type=chain_type,
            turnkey_wallet_id=existing["turnkeyWalletId"],
            turnkey_address=existing["turnkeyAddress"],
            status=existing["status"],
            created_at=existing["createdAt"],
            total_signatures=existing.get("totalSignatures", 0),
            last_used_at=existing.get("lastUsedAt"),
        )

    async def list_session_wallets(
        self,
        wallet_address: str,
        include_revoked: bool = False,
    ) -> List[SessionWalletInfo]:
        """
        List all session wallets for a user.

        Args:
            wallet_address: User's main wallet address
            include_revoked: Whether to include revoked wallets

        Returns:
            List of SessionWalletInfo
        """
        wallet_address = wallet_address.lower()

        wallets = await self._convex.query(
            "sessionWallets:listByWallet",
            {"walletAddress": wallet_address, "includeRevoked": include_revoked},
        )

        return [
            SessionWalletInfo(
                wallet_address=wallet_address,
                chain_type=w["chainType"],
                turnkey_wallet_id=w["turnkeyWalletId"],
                turnkey_address=w["turnkeyAddress"],
                status=w["status"],
                created_at=w["createdAt"],
                total_signatures=w.get("totalSignatures", 0),
                last_used_at=w.get("lastUsedAt"),
            )
            for w in (wallets or [])
        ]

    async def record_signature(self, turnkey_address: str) -> None:
        """
        Record that a signature was made with a session wallet.

        Args:
            turnkey_address: The Turnkey address that signed
        """
        try:
            await self._convex.mutation(
                "sessionWallets:recordSignature",
                {"turnkeyAddress": turnkey_address},
            )
        except Exception as e:
            # Don't fail on stats update
            logger.warning(f"Failed to record signature: {e}")

    async def revoke_session_wallet(
        self,
        turnkey_wallet_id: str,
        reason: Optional[str] = None,
    ) -> None:
        """
        Revoke a session wallet.

        Note: This only marks it as revoked in our database.
        The on-chain session key permissions must be revoked separately.

        Args:
            turnkey_wallet_id: Turnkey wallet ID to revoke
            reason: Reason for revocation
        """
        try:
            await self._convex.mutation(
                "sessionWallets:revoke",
                {"turnkeyWalletId": turnkey_wallet_id, "reason": reason},
            )
            logger.info(f"Revoked session wallet {turnkey_wallet_id}: {reason}")
        except Exception as e:
            logger.error(f"Failed to revoke session wallet: {e}")
            raise SessionWalletError(f"Failed to revoke: {e}") from e


# Singleton instance
_session_wallet_service: Optional[SessionWalletService] = None


def get_session_wallet_service() -> SessionWalletService:
    """Get the singleton session wallet service instance."""
    global _session_wallet_service
    if _session_wallet_service is None:
        _session_wallet_service = SessionWalletService()
    return _session_wallet_service
