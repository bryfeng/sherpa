import logging
from time import perf_counter
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from ..types import PortfolioResponse
from ..tools.portfolio import get_portfolio
from ..tools.defillama import get_tvl_series, get_tvl_current
from ..tools.polymarket import fetch_markets
from ..providers.coingecko import CoingeckoProvider
from ..services.trending import get_trending_tokens

router = APIRouter(prefix="/tools")
_logger = logging.getLogger(__name__)


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


@router.get("/defillama/tvl")
async def get_defillama_tvl(
    protocol: str = Query("uniswap", description="Protocol name on DefiLlama"),
    range: str = Query("7d", description="Window range: 7d or 30d"),
):
    try:
        ts, tvl = await get_tvl_series(protocol=protocol, window=range)
        return {"timestamps": ts, "tvl": tvl, "source": "defillama"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch DefiLlama TVL: {str(e)}")


@router.get("/defillama/current")
async def get_defillama_current(
    protocol: str = Query("uniswap", description="Protocol name on DefiLlama"),
):
    try:
        ts, tvl = await get_tvl_current(protocol=protocol)
        return {"timestamp": ts, "tvl": tvl, "source": "defillama"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch DefiLlama current TVL: {str(e)}")


@router.get("/polymarket/markets")
async def get_polymarket_markets(
    query: str = Query("", description="Search query"),
    limit: int = Query(5, ge=1, le=20),
):
    try:
        markets = await fetch_markets(query=query, limit=limit)
        return {"markets": markets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Polymarket markets: {str(e)}")


@router.get("/prices/top")
async def get_top_prices(
    limit: int = Query(5, ge=1, le=10),
    exclude_stable: bool = Query(True),
):
    """Return top coins by market cap with USD price and 24h change."""
    try:
        provider = CoingeckoProvider()
        if not await provider.ready():
            raise HTTPException(status_code=503, detail="Price provider unavailable")
        coins = await provider.get_top_coins(limit=limit, exclude_stable=exclude_stable)
        return {"success": True, "coins": coins}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch top prices: {str(e)}")


@router.get("/prices/trending")
async def get_trending_prices(
    limit: int = Query(10, ge=1, le=25),
):
    """Return trending tokens constrained to EVM-compatible contracts."""

    try:
        started = perf_counter()
        tokens = list(await get_trending_tokens(limit=limit))
        elapsed_ms = (perf_counter() - started) * 1000
        _logger.info(
            "trending tokens fetched", extra={
                "event": "trending_tokens",
                "limit": limit,
                "count": len(tokens),
                "duration_ms": round(elapsed_ms, 2),
            }
        )
        return {"success": True, "tokens": tokens}
    except Exception as exc:
        _logger.warning(
            "trending tokens fetch failed",
            extra={"event": "trending_tokens_error", "limit": limit, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending prices: {exc}")
