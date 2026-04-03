"""
Session Keypair Manager

Manages the backend ECDSA keypair used as the session key signer for
Rhinestone Smart Sessions. The frontend grants on-chain permission to
this keypair's address, and the backend uses it to sign intent payloads.

Called by: backend/app/api/smart_accounts.py (public key endpoint)
Called by: backend/app/providers/rhinestone.py (intent signing - future)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from eth_account import Account as EthAccount

from ...config import settings

logger = logging.getLogger(__name__)


class SessionKeypairManager:
    """
    Backend ECDSA keypair for signing intents via on-chain session keys.

    The private key is loaded from RHINESTONE_SESSION_PRIVATE_KEY env var.
    The public address is shared with the frontend so it can be included
    in the on-chain session grant (the address that's allowed to sign).
    """

    def __init__(self, private_key: str) -> None:
        if not private_key:
            raise ValueError(
                "RHINESTONE_SESSION_PRIVATE_KEY not configured. "
                "Generate one with: python -c \"from eth_account import Account; "
                "a = Account.create(); print(a.key.hex())\""
            )
        self._account = EthAccount.from_key(private_key)

    @property
    def public_address(self) -> str:
        """The checksummed address of the session keypair."""
        return self._account.address

    def sign_hash(self, message_hash: bytes) -> bytes:
        """Sign a 32-byte hash with the session key."""
        signed = self._account.signHash(message_hash)
        return bytes(signed.signature)

    def sign_message(self, message: bytes) -> bytes:
        """Sign an arbitrary message (eth_sign style)."""
        from eth_account.messages import encode_defunct

        msg = encode_defunct(primitive=message)
        signed = self._account.sign_message(msg)
        return bytes(signed.signature)


@lru_cache(maxsize=1)
def get_session_keypair_manager() -> SessionKeypairManager:
    """Singleton factory for the session keypair manager."""
    return SessionKeypairManager(settings.rhinestone_session_private_key)
