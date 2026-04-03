"""
Shared state service backed by Redis with graceful in-memory fallback.

Provides get / set / delete operations with optional TTL.
When Redis is unavailable (e.g. local dev without a Redis instance),
all operations fall back to a process-local dictionary automatically.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------

_mem_store: dict[str, Any] = {}
_mem_expiry: dict[str, float] = {}


def _mem_get(key: str) -> Optional[str]:
    if key in _mem_expiry and time.monotonic() > _mem_expiry[key]:
        _mem_store.pop(key, None)
        _mem_expiry.pop(key, None)
        return None
    return _mem_store.get(key)


def _mem_set(key: str, value: str, ttl: Optional[int] = None) -> None:
    _mem_store[key] = value
    if ttl is not None:
        _mem_expiry[key] = time.monotonic() + ttl
    else:
        _mem_expiry.pop(key, None)


def _mem_delete(key: str) -> None:
    _mem_store.pop(key, None)
    _mem_expiry.pop(key, None)


# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

_redis_client: Any = None
_redis_checked = False


def _get_redis():
    """Lazy-init a Redis client from settings.redis_url. Returns None on failure."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        from app.config import settings

        url = settings.redis_url
        if not url:
            logger.info("state: no REDIS_URL configured, using in-memory fallback")
            return None
        import redis

        _redis_client = redis.Redis.from_url(url, decode_responses=True)
        _redis_client.ping()
        logger.info("state: connected to Redis")
    except Exception as exc:
        logger.warning("state: Redis unavailable (%s), using in-memory fallback", exc)
        _redis_client = None
    return _redis_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def state_get(key: str) -> Optional[str]:
    """Return the value for *key*, or ``None`` if missing / expired."""
    r = _get_redis()
    if r is not None:
        try:
            return r.get(key)
        except Exception:
            logger.warning("state: Redis read failed for %s, falling back", key)
    return _mem_get(key)


def state_set(key: str, value: str, ttl: Optional[int] = None) -> None:
    """Store *value* under *key*.  *ttl* is expiry in seconds (optional)."""
    r = _get_redis()
    if r is not None:
        try:
            if ttl is not None:
                r.setex(key, ttl, value)
            else:
                r.set(key, value)
            return
        except Exception:
            logger.warning("state: Redis write failed for %s, falling back", key)
    _mem_set(key, value, ttl)


def state_delete(key: str) -> None:
    """Remove *key* from the store."""
    r = _get_redis()
    if r is not None:
        try:
            r.delete(key)
            return
        except Exception:
            logger.warning("state: Redis delete failed for %s, falling back", key)
    _mem_delete(key)


# ---------------------------------------------------------------------------
# JSON convenience helpers
# ---------------------------------------------------------------------------


def state_get_json(key: str) -> Any:
    """Deserialise a JSON value stored under *key*."""
    raw = state_get(key)
    if raw is None:
        return None
    return json.loads(raw)


def state_set_json(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Serialise *value* as JSON and store it under *key*."""
    state_set(key, json.dumps(value), ttl)
