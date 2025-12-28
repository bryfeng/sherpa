"""
Rate limiting middleware using Convex for storage.

Implements a sliding window rate limiter with configurable limits per endpoint.
"""

import time
from typing import Optional, Dict, Callable, Awaitable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.db.convex_client import ConvexClient, get_convex_client, ConvexError


class RateLimitExceeded(Exception):
    """Rate limit has been exceeded."""
    def __init__(self, limit: int, window_seconds: int, retry_after: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded: {limit} requests per {window_seconds}s")


# Default rate limits per endpoint pattern
DEFAULT_LIMITS: Dict[str, Dict[str, int]] = {
    # Pattern: {"limit": requests, "window": seconds}
    "/auth/nonce": {"limit": 10, "window": 60},  # 10 nonce requests per minute
    "/auth/verify": {"limit": 10, "window": 60},  # 10 verify attempts per minute
    "/chat": {"limit": 30, "window": 60},  # 30 chat requests per minute
    "/chat/stream": {"limit": 30, "window": 60},
    "/tools/portfolio": {"limit": 60, "window": 60},  # 60 portfolio requests per minute
    "/conversations": {"limit": 60, "window": 60},
    "/strategies": {"limit": 30, "window": 60},
    "default": {"limit": 100, "window": 60},  # Default: 100 requests per minute
}

# Global provider limits (shared across all users)
PROVIDER_LIMITS: Dict[str, Dict[str, int]] = {
    "alchemy": {"limit": 1000, "window": 60},  # 1000 requests per minute
    "coingecko": {"limit": 100, "window": 60},  # 100 requests per minute
    "anthropic": {"limit": 100, "window": 60},  # 100 requests per minute
}


class RateLimiter:
    """
    Rate limiter using Convex for distributed storage.

    Implements sliding window rate limiting.
    """

    def __init__(
        self,
        convex: Optional[ConvexClient] = None,
        limits: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.convex = convex or get_convex_client()
        self.limits = limits or DEFAULT_LIMITS

    def _get_limit_for_path(self, path: str) -> Dict[str, int]:
        """Get rate limit config for a path."""
        # Check for exact match first
        if path in self.limits:
            return self.limits[path]

        # Check for prefix match
        for pattern, limit in self.limits.items():
            if pattern != "default" and path.startswith(pattern):
                return limit

        return self.limits.get("default", {"limit": 100, "window": 60})

    async def check_limit(
        self,
        key: str,
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> bool:
        """
        Check if a request is within rate limits.

        Args:
            key: Unique key for this rate limit (e.g., "chat:0x123...")
            limit: Max requests allowed (uses default if not specified)
            window_seconds: Time window in seconds (uses default if not specified)

        Returns:
            True if within limits, raises RateLimitExceeded if exceeded
        """
        if limit is None:
            limit = 100
        if window_seconds is None:
            window_seconds = 60

        now = int(time.time() * 1000)  # Milliseconds for Convex
        window_start = now - (window_seconds * 1000)

        try:
            result = await self.convex.mutation(
                "rateLimit:checkAndIncrement",
                {
                    "key": key,
                    "limit": limit,
                    "windowSeconds": window_seconds,
                    "now": now,
                },
            )

            if not result.get("allowed", False):
                retry_after = int(result.get("retryAfter", window_seconds))
                raise RateLimitExceeded(limit, window_seconds, retry_after)

            return True

        except ConvexError:
            # If Convex is unavailable, allow the request (fail open)
            # In production, you might want to fail closed instead
            return True

    async def check_request(
        self,
        request: Request,
        identifier: Optional[str] = None,
    ) -> bool:
        """
        Check rate limit for an HTTP request.

        Args:
            request: FastAPI request
            identifier: Optional identifier (defaults to IP or wallet address)

        Returns:
            True if allowed, raises RateLimitExceeded if exceeded
        """
        path = request.url.path
        limit_config = self._get_limit_for_path(path)

        # Get identifier (wallet address from auth, or IP)
        if identifier is None:
            # Try to get wallet from auth
            auth = getattr(request.state, "auth", None)
            if auth and hasattr(auth, "sub"):
                identifier = auth.sub
            else:
                # Fall back to IP
                identifier = request.client.host if request.client else "unknown"

        # Create rate limit key
        key = f"{path}:{identifier}"

        return await self.check_limit(
            key=key,
            limit=limit_config["limit"],
            window_seconds=limit_config["window"],
        )

    async def check_provider_limit(self, provider: str) -> bool:
        """
        Check rate limit for an external provider (shared globally).
        """
        if provider not in PROVIDER_LIMITS:
            return True

        limit_config = PROVIDER_LIMITS[provider]
        key = f"provider:{provider}"

        return await self.check_limit(
            key=key,
            limit=limit_config["limit"],
            window_seconds=limit_config["window"],
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    """

    def __init__(
        self,
        app,
        rate_limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
        self.exclude_paths = exclude_paths or [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/healthz",
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Skip rate limiting for excluded paths
        path = request.url.path
        if any(path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        try:
            await self.rate_limiter.check_request(request)
            response = await call_next(request)
            return response

        except RateLimitExceeded as e:
            return Response(
                content=f'{{"error": "Rate limit exceeded", "retry_after": {e.retry_after}}}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(e.retry_after),
                    "X-RateLimit-Limit": str(e.limit),
                    "X-RateLimit-Window": str(e.window_seconds),
                },
            )


# Convenience function for adding middleware
def rate_limit_middleware(app, **kwargs):
    """Add rate limiting middleware to a FastAPI app."""
    return RateLimitMiddleware(app, **kwargs)


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the singleton rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
