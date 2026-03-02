"""
Cached LLM model catalog fetched from Convex.

Falls back to the hardcoded `settings.provider_models_catalog` when Convex
is unreachable or not configured.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
_cache_ts: float = 0.0
_CACHE_TTL_S = 60.0


async def get_model_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Return the model catalog, fetching from Convex with a 60s cache."""
    global _cache, _cache_ts

    if _cache is not None and (time.monotonic() - _cache_ts) < _CACHE_TTL_S:
        return _cache

    from ...config import settings

    if not settings.has_convex:
        return settings.provider_models_catalog

    try:
        from ...db.convex_client import get_convex_client

        client = get_convex_client()
        result = await client.query("admin/llmModels:getEnabledModels", {})

        if result and isinstance(result, dict):
            _cache = result
            _cache_ts = time.monotonic()
            logger.info("model_catalog_refresh", providers=list(result.keys()))
            return _cache
    except Exception:
        logger.warning("model_catalog_fetch_failed", exc_info=True)

    # Fallback to hardcoded catalog
    if _cache is not None:
        return _cache
    return settings.provider_models_catalog


def invalidate_cache() -> None:
    """Force the next call to re-fetch from Convex."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
