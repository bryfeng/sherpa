from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..types import PortfolioResponse
from ..tools.portfolio import get_portfolio

router = APIRouter(prefix="/tools")


@router.get("/portfolio")
async def get_portfolio_endpoint(
    address: str = Query(..., description="Wallet address to analyze"),
    chain: str = Query("ethereum", description="Blockchain network")
) -> PortfolioResponse:
    """Get portfolio data for a wallet address"""
    
    # Basic address validation
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid wallet address format")
    
    try:
        result = await get_portfolio(address, chain)
        
        if result.data is None:
            return PortfolioResponse(
                success=False,
                error="; ".join(result.warnings) if result.warnings else "Unknown error",
                sources=[source.model_dump() for source in result.sources]
            )
        
        return PortfolioResponse(
            success=True,
            portfolio=result.data,
            sources=[source.model_dump() for source in result.sources]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio: {str(e)}")
