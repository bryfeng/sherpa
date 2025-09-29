from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..providers.bungee import BungeeProvider

router = APIRouter(prefix="/tools/bungee")


@router.get("/quote")
async def get_bungee_quote(
    fromChainId: int = Query(..., description="Source chain ID"),
    toChainId: int = Query(..., description="Destination chain ID"),
    fromTokenAddress: str = Query(..., description="Input token address (or zero for native)"),
    toTokenAddress: str = Query(..., description="Output token address (or zero for native)"),
    amount: str = Query(..., description="Amount in smallest units (wei for 18 decimals or token decimals)"),
    userAddress: Optional[str] = Query(None, description="User EOA for the quote (required for public API)"),
    receiverAddress: Optional[str] = Query(None, description="Optional receiver (defaults to userAddress)"),
):
    if not userAddress:
        raise HTTPException(status_code=422, detail="userAddress is required for Bungee public quotes")

    params = {
        "originChainId": str(fromChainId),
        "destinationChainId": str(toChainId),
        "inputToken": fromTokenAddress,
        "outputToken": toTokenAddress,
        "inputAmount": amount,
        "userAddress": userAddress,
        "receiverAddress": receiverAddress or userAddress,
    }

    try:
        provider = BungeeProvider()
        data = await provider.quote(params)
        return {"success": True, "quote": data}
    except httpx.HTTPStatusError as e:  # type: ignore
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Bungee quote: {str(e)}")


class BuildTxRequest(BaseModel):
    quoteId: str = Field(description="Quote identifier returned by the public quote API")
    routeId: str = Field(description="Route identifier (requestHash) returned by the quote API")
    userAddress: str = Field(description="EOA that will submit the transaction")
    receiverAddress: Optional[str] = Field(default=None, description="Optional receiver (defaults to userAddress)")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Additional optional query parameters")


@router.post("/execute")
async def build_bungee_transaction(request: BuildTxRequest):
    try:
        provider = BungeeProvider()
        payload: Dict[str, Any] = {
            "quoteId": request.quoteId,
            "routeId": request.routeId,
            "userAddress": request.userAddress,
            "receiverAddress": request.receiverAddress or request.userAddress,
        }
        if request.extra:
            payload.update(request.extra)

        data = await provider.build_tx(payload)
        return {"success": True, "tx": data}
    except httpx.HTTPStatusError as e:  # type: ignore
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build Bungee transaction: {str(e)}")
