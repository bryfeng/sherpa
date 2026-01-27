"""
Smart Accounts API

Endpoints for Rhinestone Smart Account management:
- Unified multi-chain balances
- Smart Account status
- Smart Sessions management
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import settings
from ..services.unified_balance import (
    get_unified_balance_service,
    TokenBalance,
    ChainTokenBalance,
)
from ..services.gas_abstraction import get_gas_abstraction_service
from ..providers.rhinestone import get_rhinestone_provider, RhinestoneError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/smart-accounts", tags=["smart-accounts"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ChainBalanceResponse(BaseModel):
    """Token balance on a specific chain."""
    chain_id: int
    chain_name: str
    balance: str
    balance_formatted: str
    usd_value: float
    token_address: str


class TokenBalanceResponse(BaseModel):
    """Balance of a token across chains."""
    symbol: str
    name: Optional[str] = None
    total_balance: str
    total_usd_value: float
    chains: List[ChainBalanceResponse]
    logo_uri: Optional[str] = None


class UnifiedBalancesResponse(BaseModel):
    """Unified balances across all chains."""
    success: bool
    account_address: str
    total_usd_value: float
    tokens: List[TokenBalanceResponse]
    chains_queried: List[int]
    cached: bool = False
    error: Optional[str] = None


class SmartAccountStatusResponse(BaseModel):
    """Smart Account deployment status."""
    deployed: bool
    address: Optional[str] = None
    chains: List[int] = []
    modules: List[str] = []


class BestSourceChainResponse(BaseModel):
    """Best chain to source funds from."""
    success: bool
    chain_id: Optional[int] = None
    chain_name: Optional[str] = None
    available_balance: Optional[float] = None
    error: Optional[str] = None


class ChainBreakdownResponse(BaseModel):
    """USD value breakdown by chain."""
    success: bool
    breakdown: Dict[str, float]  # chain_name -> usd_value
    error: Optional[str] = None


class GasEstimateResponse(BaseModel):
    """Gas estimate for an operation."""
    operation_type: str
    estimated_usd: float
    chain_id: int
    chain_name: str
    fee_token: str = "USDC"
    fee_amount: float
    confidence: str
    breakdown: Optional[Dict[str, Any]] = None


class GasValidationResponse(BaseModel):
    """Gas validation result."""
    valid: bool
    estimated_fee_usd: float
    available_balance_usd: float
    fee_token: str
    error: Optional[str] = None


class ChainComparisonResponse(BaseModel):
    """Gas comparison across chains."""
    success: bool
    operation_type: str
    cheapest_chain_id: Optional[int] = None
    cheapest_chain_name: Optional[str] = None
    estimates: List[GasEstimateResponse]
    error: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/{account_address}/balances",
    response_model=UnifiedBalancesResponse,
    summary="Get unified multi-chain balances",
    description="Fetch token balances across all supported EVM chains for a Smart Account.",
)
async def get_unified_balances(
    account_address: str,
    chains: Optional[str] = Query(
        None,
        description="Comma-separated chain IDs (e.g., '8453,42161'). Defaults to all supported chains.",
    ),
    force_refresh: bool = Query(
        False,
        description="Skip cache and fetch fresh data",
    ),
) -> UnifiedBalancesResponse:
    """
    Get unified token balances across all chains.

    Returns aggregated balances with USD values and chain-specific breakdown.
    Results are cached for 30 seconds by default.
    """
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    # Parse chain IDs if provided
    chain_ids: Optional[List[int]] = None
    if chains:
        try:
            chain_ids = [int(c.strip()) for c in chains.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chain IDs format")

    service = get_unified_balance_service()
    result = await service.get_unified_balances(
        account_address=account_address,
        chains=chain_ids,
        force_refresh=force_refresh,
    )

    # Transform to response model
    tokens = [
        TokenBalanceResponse(
            symbol=t.symbol,
            name=t.name,
            total_balance=str(t.total_balance),
            total_usd_value=float(t.total_usd_value),
            chains=[
                ChainBalanceResponse(
                    chain_id=cb.chain_id,
                    chain_name=cb.chain_name,
                    balance=str(cb.balance),
                    balance_formatted=cb.balance_formatted,
                    usd_value=float(cb.usd_value),
                    token_address=cb.token_address,
                )
                for cb in t.chains
            ],
            logo_uri=t.logo_uri,
        )
        for t in result.tokens
    ]

    return UnifiedBalancesResponse(
        success=result.success,
        account_address=result.account_address,
        total_usd_value=float(result.total_usd_value),
        tokens=tokens,
        chains_queried=result.chains_queried,
        cached=result.cached,
        error=result.error,
    )


@router.get(
    "/{account_address}/balances/{token_symbol}",
    response_model=TokenBalanceResponse,
    summary="Get balance for a specific token",
    description="Fetch balance for a specific token across all chains.",
)
async def get_token_balance(
    account_address: str,
    token_symbol: str,
    chains: Optional[str] = Query(None, description="Comma-separated chain IDs"),
) -> TokenBalanceResponse:
    """Get balance for a specific token."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    chain_ids: Optional[List[int]] = None
    if chains:
        try:
            chain_ids = [int(c.strip()) for c in chains.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chain IDs format")

    service = get_unified_balance_service()
    token = await service.get_token_balance(
        account_address=account_address,
        token_symbol=token_symbol,
        chains=chain_ids,
    )

    if not token:
        raise HTTPException(status_code=404, detail=f"Token {token_symbol} not found")

    return TokenBalanceResponse(
        symbol=token.symbol,
        name=token.name,
        total_balance=str(token.total_balance),
        total_usd_value=float(token.total_usd_value),
        chains=[
            ChainBalanceResponse(
                chain_id=cb.chain_id,
                chain_name=cb.chain_name,
                balance=str(cb.balance),
                balance_formatted=cb.balance_formatted,
                usd_value=float(cb.usd_value),
                token_address=cb.token_address,
            )
            for cb in token.chains
        ],
        logo_uri=token.logo_uri,
    )


@router.get(
    "/{account_address}/chain-breakdown",
    response_model=ChainBreakdownResponse,
    summary="Get USD value breakdown by chain",
    description="Get total USD value on each chain.",
)
async def get_chain_breakdown(
    account_address: str,
) -> ChainBreakdownResponse:
    """Get USD value breakdown by chain."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    service = get_unified_balance_service()
    breakdown = await service.get_chain_breakdown(account_address)

    # Convert chain IDs to names
    from ..services.unified_balance import SUPPORTED_CHAINS

    named_breakdown = {
        SUPPORTED_CHAINS.get(chain_id, f"Chain {chain_id}"): float(usd_value)
        for chain_id, usd_value in breakdown.items()
    }

    return ChainBreakdownResponse(
        success=True,
        breakdown=named_breakdown,
    )


@router.get(
    "/{account_address}/best-source-chain",
    response_model=BestSourceChainResponse,
    summary="Find best chain to source funds",
    description="Find the optimal chain to source funds for an intent.",
)
async def get_best_source_chain(
    account_address: str,
    token_symbol: str = Query(..., description="Token to source (e.g., 'USDC')"),
    amount_usd: float = Query(..., description="Required USD value"),
) -> BestSourceChainResponse:
    """Find the best chain to source funds from."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    service = get_unified_balance_service()
    chain_id = await service.find_best_source_chain(
        account_address=account_address,
        token_symbol=token_symbol,
        amount_usd=Decimal(str(amount_usd)),
    )

    if not chain_id:
        return BestSourceChainResponse(
            success=False,
            error=f"Insufficient {token_symbol} balance for ${amount_usd}",
        )

    from ..services.unified_balance import SUPPORTED_CHAINS

    # Get the available balance on this chain
    token = await service.get_token_balance(account_address, token_symbol)
    available_balance = None
    if token:
        for cb in token.chains:
            if cb.chain_id == chain_id:
                available_balance = float(cb.usd_value)
                break

    return BestSourceChainResponse(
        success=True,
        chain_id=chain_id,
        chain_name=SUPPORTED_CHAINS.get(chain_id, f"Chain {chain_id}"),
        available_balance=available_balance,
    )


@router.get(
    "/{owner_address}/status",
    response_model=SmartAccountStatusResponse,
    summary="Get Smart Account status",
    description="Check if an owner has a deployed Smart Account.",
)
async def get_smart_account_status(
    owner_address: str,
) -> SmartAccountStatusResponse:
    """Check Smart Account deployment status."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    try:
        provider = get_rhinestone_provider()
        status = await provider.get_smart_account_status(owner_address)

        return SmartAccountStatusResponse(
            deployed=status.get("deployed", False),
            address=status.get("address"),
            chains=status.get("chains", []),
            modules=status.get("modules", []),
        )
    except RhinestoneError as e:
        logger.error(f"Failed to get Smart Account status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{account_address}/balances/refresh",
    response_model=UnifiedBalancesResponse,
    summary="Refresh cached balances",
    description="Force refresh cached balances for an account.",
)
async def refresh_balances(
    account_address: str,
) -> UnifiedBalancesResponse:
    """Force refresh balances cache."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    service = get_unified_balance_service()

    # Clear cache for this address
    service.clear_cache(account_address)

    # Fetch fresh data
    result = await service.get_unified_balances(
        account_address=account_address,
        force_refresh=True,
    )

    tokens = [
        TokenBalanceResponse(
            symbol=t.symbol,
            name=t.name,
            total_balance=str(t.total_balance),
            total_usd_value=float(t.total_usd_value),
            chains=[
                ChainBalanceResponse(
                    chain_id=cb.chain_id,
                    chain_name=cb.chain_name,
                    balance=str(cb.balance),
                    balance_formatted=cb.balance_formatted,
                    usd_value=float(cb.usd_value),
                    token_address=cb.token_address,
                )
                for cb in t.chains
            ],
            logo_uri=t.logo_uri,
        )
        for t in result.tokens
    ]

    return UnifiedBalancesResponse(
        success=result.success,
        account_address=result.account_address,
        total_usd_value=float(result.total_usd_value),
        tokens=tokens,
        chains_queried=result.chains_queried,
        cached=False,
        error=result.error,
    )


# ============================================================================
# Gas Estimation Endpoints
# ============================================================================


@router.get(
    "/gas/estimate",
    response_model=GasEstimateResponse,
    summary="Estimate gas cost for an operation",
    description="Get estimated USDC cost for gas on an intent operation.",
)
async def estimate_gas(
    operation_type: str = Query(
        ...,
        description="Operation type: swap, bridge, transfer, approve, dca_execution, copy_trade",
    ),
    target_chain: int = Query(..., description="Target chain ID"),
    source_chains: Optional[str] = Query(
        None,
        description="Comma-separated source chain IDs for cross-chain intents",
    ),
) -> GasEstimateResponse:
    """Estimate gas cost for an intent operation."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    # Parse source chains
    source_chain_ids: Optional[List[int]] = None
    if source_chains:
        try:
            source_chain_ids = [int(c.strip()) for c in source_chains.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid source chain IDs format")

    service = get_gas_abstraction_service()
    estimate = await service.estimate_intent_gas(
        operation_type=operation_type,
        target_chain=target_chain,
        source_chains=source_chain_ids,
        cross_chain=bool(source_chain_ids and len(source_chain_ids) > 1),
    )

    return GasEstimateResponse(
        operation_type=estimate.operation_type,
        estimated_usd=float(estimate.estimated_usd),
        chain_id=estimate.chain_id,
        chain_name=estimate.chain_name,
        fee_token=estimate.fee_token,
        fee_amount=float(estimate.fee_amount),
        confidence=estimate.confidence,
        breakdown=estimate.breakdown,
    )


@router.get(
    "/{account_address}/gas/validate",
    response_model=GasValidationResponse,
    summary="Validate gas balance",
    description="Check if account has sufficient USDC for gas.",
)
async def validate_gas_balance(
    account_address: str,
    operation_type: str = Query(..., description="Operation type"),
    target_chain: int = Query(..., description="Target chain ID"),
    source_chains: Optional[str] = Query(None, description="Comma-separated source chain IDs"),
) -> GasValidationResponse:
    """Validate that account has sufficient USDC for gas."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    source_chain_ids: Optional[List[int]] = None
    if source_chains:
        try:
            source_chain_ids = [int(c.strip()) for c in source_chains.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid source chain IDs format")

    service = get_gas_abstraction_service()
    result = await service.validate_gas_balance(
        account_address=account_address,
        operation_type=operation_type,
        target_chain=target_chain,
        source_chains=source_chain_ids,
    )

    return GasValidationResponse(
        valid=result.valid,
        estimated_fee_usd=float(result.estimated_fee_usd),
        available_balance_usd=float(result.available_balance_usd),
        fee_token=result.fee_token,
        error=result.error,
    )


@router.get(
    "/gas/compare-chains",
    response_model=ChainComparisonResponse,
    summary="Compare gas costs across chains",
    description="Find the cheapest chain for an operation type.",
)
async def compare_chain_gas(
    operation_type: str = Query(..., description="Operation type"),
    chains: Optional[str] = Query(None, description="Comma-separated chain IDs to compare"),
) -> ChainComparisonResponse:
    """Compare gas costs across chains."""
    if not settings.enable_rhinestone:
        raise HTTPException(status_code=503, detail="Rhinestone not enabled")

    chain_ids: Optional[List[int]] = None
    if chains:
        try:
            chain_ids = [int(c.strip()) for c in chains.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chain IDs format")

    service = get_gas_abstraction_service()
    estimates = await service.compare_chains(
        operation_type=operation_type,
        chains=chain_ids,
    )

    estimate_responses = [
        GasEstimateResponse(
            operation_type=e.operation_type,
            estimated_usd=float(e.estimated_usd),
            chain_id=e.chain_id,
            chain_name=e.chain_name,
            fee_token=e.fee_token,
            fee_amount=float(e.fee_amount),
            confidence=e.confidence,
            breakdown=e.breakdown,
        )
        for e in estimates
    ]

    cheapest = estimates[0] if estimates else None

    return ChainComparisonResponse(
        success=True,
        operation_type=operation_type,
        cheapest_chain_id=cheapest.chain_id if cheapest else None,
        cheapest_chain_name=cheapest.chain_name if cheapest else None,
        estimates=estimate_responses,
    )
