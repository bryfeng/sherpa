"""
Authentication service using wallet sign-in (EVM SIWE + Solana sign-in).
"""

import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional

from siwe import SiweMessage, VerificationError

from app.config import settings
from app.db.convex_client import ConvexClient, get_convex_client
from app.services.address import normalize_chain

from .models import (
    AuthSession,
    VerifiedWallet,
    AuthError,
    SessionExpiredError,
    InvalidSignatureError,
    InvalidNonceError,
    Scope,
    TokenPayload,
)
from .solana_signin import parse_solana_signin_message, verify_solana_signature


# JWT configuration - MUST be set in production
if not settings.convex_internal_api_key:
    raise ValueError(
        "CONVEX_INTERNAL_API_KEY must be set for JWT signing. "
        "This is required for authentication security."
    )
JWT_SECRET = settings.convex_internal_api_key
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 7
NONCE_EXPIRE_MINUTES = 10


class AuthService:
    """
    Authentication service using wallet sign-in.

    Supports:
    - EVM wallets via SIWE (Sign-In With Ethereum)
    - Solana wallets via Sign-In with Solana message format

    Flow:
    1. Client requests nonce via POST /auth/nonce
    2. Client signs message with wallet
    3. Client sends message + signature + chain to POST /auth/verify
    4. Server verifies and returns JWT tokens
    5. Client uses access token for authenticated requests
    6. Client refreshes token via POST /auth/refresh
    """

    def __init__(self, convex: Optional[ConvexClient] = None):
        self.convex = convex or get_convex_client()

    async def generate_nonce(self, wallet_address: str, chain: Optional[str] = None) -> dict:
        """
        Generate a unique nonce for wallet sign-in.

        The nonce is stored in Convex and expires after 10 minutes.
        """
        chain_slug = normalize_chain(chain) if chain else None
        normalized_address = self._normalize_wallet_address(wallet_address, chain_slug)
        # SIWE spec (EIP-4361) requires alphanumeric nonce only - no special characters
        # Use token_hex which generates hex characters (0-9, a-f)
        nonce = secrets.token_hex(16)
        expires_at = datetime.utcnow() + timedelta(minutes=NONCE_EXPIRE_MINUTES)

        # Store nonce in Convex
        await self.convex.mutation(
            "auth:createNonce",
            {
                "walletAddress": normalized_address,
                "nonce": nonce,
                "expiresAt": int(expires_at.timestamp() * 1000),
            },
        )

        return {
            "nonce": nonce,
            "expires_at": expires_at.isoformat(),
        }

    async def verify_signature(
        self,
        message: str,
        signature: str,
        chain: Optional[str] = None,
    ) -> VerifiedWallet:
        """
        Verify a wallet signature and return the verified wallet.
        """
        chain_slug = normalize_chain(chain) if chain else None
        if chain_slug == "solana":
            return await self._verify_solana_signature(message, signature)
        return await self._verify_evm_signature(message, signature)

    async def create_session(
        self,
        wallet: VerifiedWallet,
        scopes: Optional[list] = None,
    ) -> AuthSession:
        """
        Create a new authenticated session for a verified wallet.

        This also creates/gets the user and wallet in Convex.
        """
        # Get or create user in Convex
        chain_slug = self._chain_slug_from_id(wallet.chain_id)
        result = await self.convex.get_or_create_user(
            address=wallet.address,
            chain=chain_slug,
        )

        user_id = result.get("user", {}).get("_id")
        wallet_id = result.get("wallet", {}).get("_id")

        # Generate session ID
        session_id = secrets.token_urlsafe(32)

        # Default scopes
        if scopes is None:
            scopes = [Scope.READ_PORTFOLIO, Scope.READ_HISTORY]

        # Generate tokens
        access_token = self._generate_access_token(
            wallet_address=wallet.address,
            session_id=session_id,
            chain_id=wallet.chain_id,
            user_id=user_id,
            wallet_id=wallet_id,
            scopes=scopes,
        )

        refresh_token = self._generate_refresh_token(
            wallet_address=wallet.address,
            session_id=session_id,
        )

        expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        # Store session in Convex
        await self.convex.mutation(
            "auth:createSession",
            {
                "sessionId": session_id,
                "walletAddress": wallet.address,
                "chainId": wallet.chain_id,
                "userId": user_id,
                "walletId": wallet_id,
                "expiresAt": int(expires_at.timestamp() * 1000),
                "scopes": [s.value if isinstance(s, Scope) else s for s in scopes],
            },
        )

        return AuthSession(
            session_id=session_id,
            wallet_address=wallet.address,
            chain_id=wallet.chain_id,
            user_id=user_id,
            wallet_id=wallet_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scopes=scopes,
        )

    async def refresh_session(self, refresh_token: str) -> AuthSession:
        """
        Refresh an existing session using a refresh token.
        """
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            if payload.get("type") != "refresh":
                raise AuthError("Invalid token type")

            session_id = payload.get("session_id")
            wallet_address = payload.get("sub")

            # Verify session still exists in Convex
            session = await self.convex.query(
                "auth:getSession",
                {"sessionId": session_id},
            )

            if not session:
                raise SessionExpiredError("Session not found")

            # Generate new tokens
            new_session_id = secrets.token_urlsafe(32)
            scopes = [Scope(s) for s in session.get("scopes", [])]

            access_token = self._generate_access_token(
                wallet_address=wallet_address,
                session_id=new_session_id,
                chain_id=session.get("chainId", 1),
                user_id=session.get("userId"),
                wallet_id=session.get("walletId"),
                scopes=scopes,
            )

            new_refresh_token = self._generate_refresh_token(
                wallet_address=wallet_address,
                session_id=new_session_id,
            )

            expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

            # Update session in Convex
            await self.convex.mutation(
                "auth:updateSession",
                {
                    "oldSessionId": session_id,
                    "newSessionId": new_session_id,
                    "expiresAt": int(expires_at.timestamp() * 1000),
                },
            )

            return AuthSession(
                session_id=new_session_id,
                wallet_address=wallet_address,
                chain_id=session.get("chainId", 1),
                user_id=session.get("userId"),
                wallet_id=session.get("walletId"),
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_at=expires_at,
                scopes=scopes,
            )

        except jwt.ExpiredSignatureError:
            raise SessionExpiredError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthError(f"Invalid refresh token: {e}")

    async def verify_access_token(self, token: str) -> TokenPayload:
        """
        Verify an access token and return the payload.
        """
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            if payload.get("type") != "access":
                raise AuthError("Invalid token type")

            return TokenPayload(
                sub=payload["sub"],
                session_id=payload["session_id"],
                chain_id=payload.get("chain_id", 1),
                user_id=payload.get("user_id"),
                wallet_id=payload.get("wallet_id"),
                scopes=payload.get("scopes", []),
                exp=payload["exp"],
                iat=payload["iat"],
                type="access",
            )

        except jwt.ExpiredSignatureError:
            raise SessionExpiredError("Access token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthError(f"Invalid access token: {e}")

    async def revoke_session(self, session_id: str) -> None:
        """
        Revoke a session (logout).
        """
        await self.convex.mutation(
            "auth:revokeSession",
            {"sessionId": session_id},
        )

    def _generate_access_token(
        self,
        wallet_address: str,
        session_id: str,
        chain_id: int | str,
        user_id: Optional[str],
        wallet_id: Optional[str],
        scopes: list,
    ) -> str:
        """Generate a JWT access token."""
        now = datetime.utcnow()
        payload = {
            "sub": wallet_address,
            "session_id": session_id,
            "chain_id": chain_id,
            "user_id": user_id,
            "wallet_id": wallet_id,
            "scopes": [s.value if isinstance(s, Scope) else s for s in scopes],
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": now,
            "type": "access",
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def _generate_refresh_token(
        self,
        wallet_address: str,
        session_id: str,
    ) -> str:
        """Generate a JWT refresh token."""
        now = datetime.utcnow()
        payload = {
            "sub": wallet_address,
            "session_id": session_id,
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": now,
            "type": "refresh",
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    async def _verify_nonce(self, wallet_address: str, nonce: str) -> bool:
        """Verify a nonce exists and hasn't expired."""
        result = await self.convex.query(
            "auth:verifyNonce",
            {
                "walletAddress": wallet_address,
                "nonce": nonce,
            },
        )
        return result is True

    async def _invalidate_nonce(self, wallet_address: str, nonce: str) -> None:
        """Invalidate a nonce after use."""
        await self.convex.mutation(
            "auth:deleteNonce",
            {
                "walletAddress": wallet_address,
                "nonce": nonce,
            },
        )

    async def _verify_evm_signature(
        self,
        message: str,
        signature: str,
    ) -> VerifiedWallet:
        try:
            siwe_message = SiweMessage.from_message(message)
            siwe_message.verify(signature)

            normalized_address = self._normalize_wallet_address(
                siwe_message.address,
                "ethereum",
            )

            nonce_valid = await self._verify_nonce(
                normalized_address,
                siwe_message.nonce,
            )
            if not nonce_valid:
                raise InvalidNonceError("Nonce is invalid or expired")

            await self._invalidate_nonce(normalized_address, siwe_message.nonce)

            return VerifiedWallet(
                address=normalized_address,
                chain_id=siwe_message.chain_id,
            )

        except VerificationError as e:
            raise InvalidSignatureError(f"Signature verification failed: {e}")

    async def _verify_solana_signature(
        self,
        message: str,
        signature: str,
    ) -> VerifiedWallet:
        try:
            parsed = parse_solana_signin_message(message)
        except ValueError as e:
            raise InvalidSignatureError(str(e))
        normalized_address = self._normalize_wallet_address(parsed.address, "solana")
        try:
            verify_solana_signature(message, signature, normalized_address)
        except ValueError as e:
            raise InvalidSignatureError(str(e))

        nonce_valid = await self._verify_nonce(normalized_address, parsed.nonce)
        if not nonce_valid:
            raise InvalidNonceError("Nonce is invalid or expired")

        await self._invalidate_nonce(normalized_address, parsed.nonce)

        return VerifiedWallet(
            address=normalized_address,
            chain_id="solana",
        )

    def _normalize_wallet_address(self, address: str, chain_slug: Optional[str]) -> str:
        if not address:
            return address
        if chain_slug == "solana":
            return address
        if chain_slug:
            return address.lower()
        if address.startswith("0x"):
            return address.lower()
        return address

    def _chain_slug_from_id(self, chain_id: int | str) -> str:
        if isinstance(chain_id, str):
            if chain_id.lower() == "solana":
                return "solana"
            return chain_id
        if chain_id == 1:
            return "ethereum"
        return f"chain:{chain_id}"


# Singleton instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get the singleton auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
