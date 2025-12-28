"""
FastAPI authentication middleware and dependencies.
"""

from typing import Optional, List
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .service import AuthService, get_auth_service
from .models import TokenPayload, AuthError, SessionExpiredError, Scope


# HTTP Bearer token scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_wallet(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[TokenPayload]:
    """
    Get the current authenticated wallet from the request.

    Returns None if no valid token is provided.
    Use `require_auth` for endpoints that require authentication.
    """
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        payload = await auth_service.verify_access_token(token)
        return payload
    except (AuthError, SessionExpiredError):
        return None


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenPayload:
    """
    Require authentication for an endpoint.

    Raises HTTPException 401 if not authenticated.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token = credentials.credentials
        payload = await auth_service.verify_access_token(token)
        return payload
    except SessionExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_scopes(*required_scopes: Scope):
    """
    Factory for creating scope-checking dependencies.

    Usage:
        @app.get("/strategies")
        async def list_strategies(
            auth: TokenPayload = Depends(require_scopes(Scope.MANAGE_STRATEGIES))
        ):
            ...
    """
    async def check_scopes(
        auth: TokenPayload = Depends(require_auth),
    ) -> TokenPayload:
        # Check if user has required scopes
        user_scopes = set(auth.scopes)
        required = set(s.value for s in required_scopes)

        # Admin has all scopes
        if Scope.ADMIN.value in user_scopes:
            return auth

        if not required.issubset(user_scopes):
            missing = required - user_scopes
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {', '.join(missing)}",
            )

        return auth

    return check_scopes


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[TokenPayload]:
    """
    Optional authentication - returns None if not authenticated.

    Use this for endpoints that work differently for authenticated users.
    """
    return await get_current_wallet(credentials, auth_service)


class AuthMiddleware:
    """
    Middleware to add auth context to requests.

    This allows accessing the current user in any part of the request lifecycle.
    """

    def __init__(self, app):
        self.app = app
        self.auth_service = get_auth_service()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract token from headers
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    payload = await self.auth_service.verify_access_token(token)
                    scope["auth"] = payload
                except (AuthError, SessionExpiredError):
                    scope["auth"] = None
            else:
                scope["auth"] = None

        await self.app(scope, receive, send)


def get_auth_from_request(request: Request) -> Optional[TokenPayload]:
    """
    Get auth payload from request scope.

    Only works if AuthMiddleware is installed.
    """
    return getattr(request.scope, "auth", None)
