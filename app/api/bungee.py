from fastapi import APIRouter, HTTPException, Query
import httpx
from typing import Optional
from ..providers.bungee import BungeeProvider

router = APIRouter(prefix="/tools/bungee")


@router.get("/quote")
async def get_bungee_quote(
    fromChainId: int = Query(..., description="Source chain ID"),
    toChainId: int = Query(..., description="Destination chain ID"),
    fromTokenAddress: str = Query(..., description="Token address on source chain (or native placeholder)"),
    toTokenAddress: str = Query(..., description="Token address on dest chain (or native placeholder)"),
    amount: str = Query(..., description="Amount in smallest units (wei for 18 decimals)"),
    userAddress: Optional[str] = Query(None, description="User EOA for route estimation"),
    slippage: Optional[float] = Query(1.0, description="Max slippage percent"),
):
    try:
        provider = BungeeProvider()
        payload = {
            "fromChainId": fromChainId,
            "toChainId": toChainId,
            "fromTokenAddress": fromTokenAddress,
            "toTokenAddress": toTokenAddress,
            "amount": amount,
            "userAddress": userAddress,
            "slippage": slippage,
        }
        data = await provider.quote({k: v for k, v in payload.items() if v is not None})
        return {"success": True, "quote": data}
    except httpx.HTTPStatusError as e:  # type: ignore
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Bungee quote: {str(e)}")
