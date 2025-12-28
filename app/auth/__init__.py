from .service import AuthService, get_auth_service
from .models import (
    AuthSession,
    VerifiedWallet,
    AuthError,
    SessionExpiredError,
    TokenPayload,
    Scope,
)
from .middleware import (
    require_auth,
    optional_auth,
    require_scopes,
    get_current_wallet,
    AuthMiddleware,
)

__all__ = [
    "AuthService",
    "get_auth_service",
    "AuthSession",
    "VerifiedWallet",
    "AuthError",
    "SessionExpiredError",
    "TokenPayload",
    "Scope",
    "require_auth",
    "optional_auth",
    "require_scopes",
    "get_current_wallet",
    "AuthMiddleware",
]
