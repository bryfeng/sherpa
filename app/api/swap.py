from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from ..tools.aggregators import quote_swap_simple


router = APIRouter(prefix="/swap")


class SwapQuoteRequest(BaseModel):
    token_in: str = Field(description="Symbol or address of input token (MVP uses symbols)")
    token_out: str = Field(description="Symbol or address of output token (MVP uses symbols)")
    amount_in: float = Field(gt=0, description="Amount of input token")
    chain: Optional[str] = Field(default="ethereum", description="Blockchain network context")
    slippage_bps: Optional[int] = Field(default=50, ge=0, le=5000, description="Allowed slippage in basis points")


class SwapQuoteResponse(BaseModel):
    success: bool
    from_token: str
    to_token: str
    amount_in: float
    amount_out_est: float
    price_in_usd: float
    price_out_usd: float
    fee_est: float
    slippage_bps: int
    route: Dict[str, Any]
    sources: list
    warnings: list = []


@router.post("/quote")
async def post_swap_quote(req: SwapQuoteRequest) -> SwapQuoteResponse:
    try:
        quote = await quote_swap_simple(
            token_in=req.token_in,
            token_out=req.token_out,
            amount_in=req.amount_in,
            slippage_bps=req.slippage_bps or 50,
        )
        return SwapQuoteResponse(
            success=bool(quote.get("success", False)),
            from_token=str(quote.get("from")),
            to_token=str(quote.get("to")),
            amount_in=float(quote.get("amount_in", 0)),
            amount_out_est=float(quote.get("amount_out_est", 0)),
            price_in_usd=float(quote.get("price_in_usd", 0)),
            price_out_usd=float(quote.get("price_out_usd", 0)),
            fee_est=float(quote.get("fee_est", 0)),
            slippage_bps=int(quote.get("slippage_bps", req.slippage_bps or 50)),
            route=dict(quote.get("route", {})),
            sources=list(quote.get("sources", [])),
            warnings=list(quote.get("warnings", [])),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute quote: {str(e)}")

