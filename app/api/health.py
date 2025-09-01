from fastapi import APIRouter
from typing import Dict, Any
from ..providers.alchemy import AlchemyProvider
from ..providers.coingecko import CoingeckoProvider

router = APIRouter()


@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint that verifies provider status"""
    
    # Initialize providers
    alchemy = AlchemyProvider()
    coingecko = CoingeckoProvider()
    
    # Check each provider
    provider_status = {}
    
    # Check Alchemy
    provider_status["alchemy"] = await alchemy.health_check()
    
    # Check Coingecko  
    provider_status["coingecko"] = await coingecko.health_check()
    
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
