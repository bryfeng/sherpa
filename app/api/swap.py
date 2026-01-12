from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from ..tools.aggregators import quote_swap_simple


router = APIRouter(prefix="/swap")


class SwapQuoteRequest(BaseModel):
    token_in: str = Field(description="Symbol or address of input token")
    token_out: str = Field(description="Symbol or address of output token")
    amount_in: Optional[float] = Field(default=None, gt=0, description="Amount of input token (use this OR amount_usd)")
    amount_usd: Optional[float] = Field(default=None, gt=0, description="Amount in USD to spend (converts to token amount)")
    chain: Optional[str] = Field(default="ethereum", description="Blockchain network context")
    slippage_bps: Optional[int] = Field(default=50, ge=0, le=5000, description="Allowed slippage in basis points")
    wallet_address: Optional[str] = Field(
        default=None,
        description="Wallet address used for Relay quoting (required for executable quotes)",
    )


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


# Stablecoins that are pegged 1:1 to USD
STABLECOINS = {"usdc", "usdt", "dai", "busd", "tusd", "usdp", "gusd", "frax"}


@router.post("/quote")
async def post_swap_quote(req: SwapQuoteRequest) -> SwapQuoteResponse:
    # Validate that we have either amount_in or amount_usd
    if req.amount_in is None and req.amount_usd is None:
        raise HTTPException(status_code=400, detail="Either amount_in or amount_usd must be provided")

    # Determine the actual amount_in to use
    amount_in = req.amount_in
    if amount_in is None and req.amount_usd is not None:
        # Convert USD to token amount
        token_symbol = req.token_in.lower()

        if token_symbol in STABLECOINS:
            # For stablecoins, 1 token ≈ $1 USD
            amount_in = req.amount_usd
        else:
            # For non-stablecoins, we need to fetch price first
            # Do a small quote to get the price, then calculate the amount
            try:
                # Quote for 1 token to get price
                test_quote = await quote_swap_simple(
                    token_in=req.token_in,
                    token_out=req.token_out,
                    amount_in=0.001,  # Small amount to get price
                    slippage_bps=req.slippage_bps or 50,
                    chain=req.chain,
                    wallet_address=req.wallet_address,
                )
                price_per_token = test_quote.get("price_in_usd", 0)
                if price_per_token and price_per_token > 0:
                    # Convert USD to token amount
                    amount_in = req.amount_usd / price_per_token
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not determine price for {req.token_in}. Please use amount_in instead."
                    )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Failed to get price for conversion: {exc}")

    try:
        quote = await quote_swap_simple(
            token_in=req.token_in,
            token_out=req.token_out,
            amount_in=amount_in,
            slippage_bps=req.slippage_bps or 50,
            chain=req.chain,
            wallet_address=req.wallet_address,
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute quote: {exc}")
