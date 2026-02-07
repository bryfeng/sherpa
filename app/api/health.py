from fastapi import APIRouter
from typing import Dict, Any
from ..providers.alchemy import AlchemyProvider
from ..providers.coingecko import CoingeckoProvider
from ..config import settings

router = APIRouter()


@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint that verifies provider status"""

    # Initialize providers
    alchemy = AlchemyProvider()
    coingecko = CoingeckoProvider()

    # Check each provider
    provider_status: Dict[str, Any] = {}

    # Check Alchemy
    provider_status["alchemy"] = await alchemy.health_check()

    # Check Coingecko
    provider_status["coingecko"] = await coingecko.health_check()

    # Check Rhinestone availability
    try:
        from ..core.strategies.dca.executor import RHINESTONE_AVAILABLE
        provider_status["rhinestone"] = {
            "status": "healthy" if (settings.enable_rhinestone and RHINESTONE_AVAILABLE) else "unavailable",
            "enabled": settings.enable_rhinestone,
            "sdk_available": RHINESTONE_AVAILABLE,
            "has_api_key": bool(settings.rhinestone_api_key),
            "has_session_key": bool(settings.rhinestone_session_private_key),
        }
    except Exception:
        provider_status["rhinestone"] = {"status": "unavailable", "enabled": False}

    # Determine overall health
    all_healthy = all(
        status["status"] in ["healthy", "unavailable"]
        for status in provider_status.values()
    )

    # Count available providers
    available_providers = sum(
        1 for status in provider_status.values()
        if status["status"] == "healthy"
    )

    return {
        "status": "healthy" if all_healthy and available_providers > 0 else "degraded",
        "providers": provider_status,
        "available_providers": available_providers,
        "total_providers": len(provider_status)
    }
