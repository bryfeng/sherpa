"""
DCA Strategy API Endpoints

REST API for managing DCA (Dollar Cost Averaging) strategies.
"""

from decimal import Decimal
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from app.core.strategies.dca import (
    DCAService,
    DCAStrategy,
    DCAConfig,
    DCAExecution,
    DCAFrequency,
    DCAStatus,
    TokenInfo,
)
from app.db.convex_client import get_convex_client
from app.config import settings

router = APIRouter(prefix="/dca", tags=["DCA Strategies"])


# =============================================================================
# Request/Response Models
# =============================================================================


class TokenInfoRequest(BaseModel):
    """Token information for DCA."""
    symbol: str = Field(..., description="Token symbol (e.g., ETH)")
    address: str = Field(..., description="Token contract address")
    chain_id: int = Field(..., alias="chainId", description="Chain ID")
    decimals: int = Field(..., description="Token decimals")

    class Config:
        populate_by_name = True


class CreateDCARequest(BaseModel):
    """Request to create a new DCA strategy."""
    user_id: str = Field(..., alias="userId", description="User ID")
    wallet_id: str = Field(..., alias="walletId", description="Wallet ID")
    wallet_address: str = Field(..., alias="walletAddress", description="Wallet address")
    name: str = Field(..., description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    from_token: TokenInfoRequest = Field(..., alias="fromToken")
    to_token: TokenInfoRequest = Field(..., alias="toToken")
    amount_per_execution_usd: float = Field(
        ..., alias="amountPerExecutionUsd", gt=0, description="USD amount per execution"
    )
    frequency: DCAFrequency = Field(..., description="Execution frequency")
    execution_hour_utc: int = Field(
        9, alias="executionHourUtc", ge=0, le=23, description="Hour of day (UTC)"
    )
    execution_day_of_week: Optional[int] = Field(
        None, alias="executionDayOfWeek", ge=0, le=6, description="Day of week (0=Mon)"
    )
    execution_day_of_month: Optional[int] = Field(
        None, alias="executionDayOfMonth", ge=1, le=31, description="Day of month"
    )
    cron_expression: Optional[str] = Field(
        None, alias="cronExpression", description="Custom cron expression"
    )
    max_slippage_bps: int = Field(
        100, alias="maxSlippageBps", ge=0, le=1000, description="Max slippage in basis points"
    )
    max_gas_usd: float = Field(
        10, alias="maxGasUsd", gt=0, description="Max gas cost in USD"
    )
    skip_if_gas_above_usd: Optional[float] = Field(
        None, alias="skipIfGasAboveUsd", description="Skip if gas exceeds this"
    )
    pause_if_price_above_usd: Optional[float] = Field(
        None, alias="pauseIfPriceAboveUsd", description="Pause if price exceeds this"
    )
    pause_if_price_below_usd: Optional[float] = Field(
        None, alias="pauseIfPriceBelowUsd", description="Pause if price below this"
    )
    max_total_spend_usd: Optional[float] = Field(
        None, alias="maxTotalSpendUsd", description="Max total spend limit"
    )
    max_executions: Optional[int] = Field(
        None, alias="maxExecutions", description="Max number of executions"
    )
    end_date: Optional[datetime] = Field(
        None, alias="endDate", description="Strategy end date"
    )

    class Config:
        populate_by_name = True


class ActivateRequest(BaseModel):
    """Request to activate a DCA strategy."""
    session_key_id: str = Field(..., alias="sessionKeyId", description="Session key ID")

    class Config:
        populate_by_name = True


class UpdateConfigRequest(BaseModel):
    """Request to update DCA configuration."""
    amount_per_execution_usd: Optional[float] = Field(None, alias="amountPerExecutionUsd")
    frequency: Optional[DCAFrequency] = None
    execution_hour_utc: Optional[int] = Field(None, alias="executionHourUtc")
    execution_day_of_week: Optional[int] = Field(None, alias="executionDayOfWeek")
    execution_day_of_month: Optional[int] = Field(None, alias="executionDayOfMonth")
    cron_expression: Optional[str] = Field(None, alias="cronExpression")
    max_slippage_bps: Optional[int] = Field(None, alias="maxSlippageBps")
    max_gas_usd: Optional[float] = Field(None, alias="maxGasUsd")
    skip_if_gas_above_usd: Optional[float] = Field(None, alias="skipIfGasAboveUsd")
    pause_if_price_above_usd: Optional[float] = Field(None, alias="pauseIfPriceAboveUsd")
    pause_if_price_below_usd: Optional[float] = Field(None, alias="pauseIfPriceBelowUsd")
    max_total_spend_usd: Optional[float] = Field(None, alias="maxTotalSpendUsd")
    max_executions: Optional[int] = Field(None, alias="maxExecutions")
    end_date: Optional[datetime] = Field(None, alias="endDate")

    class Config:
        populate_by_name = True


class DCAStrategyResponse(BaseModel):
    """DCA strategy response."""
    id: str
    user_id: str = Field(..., alias="userId")
    wallet_id: str = Field(..., alias="walletId")
    wallet_address: str = Field(..., alias="walletAddress")
    name: str
    description: Optional[str] = None
    status: DCAStatus
    from_token: dict = Field(..., alias="fromToken")
    to_token: dict = Field(..., alias="toToken")
    amount_per_execution_usd: float = Field(..., alias="amountPerExecutionUsd")
    frequency: DCAFrequency
    schedule_description: str = Field(..., alias="scheduleDescription")
    next_execution_at: Optional[datetime] = Field(None, alias="nextExecutionAt")
    last_execution_at: Optional[datetime] = Field(None, alias="lastExecutionAt")
    stats: dict
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    class Config:
        populate_by_name = True


class SessionKeyRequirementsResponse(BaseModel):
    """Session key requirements for a DCA strategy."""
    permissions: List[str]
    value_per_tx_usd: float = Field(..., alias="valuePerTxUsd")
    total_value_usd: float = Field(..., alias="totalValueUsd")
    token_allowlist: List[str] = Field(..., alias="tokenAllowlist")
    chain_allowlist: List[int] = Field(..., alias="chainAllowlist")
    duration_days: int = Field(..., alias="durationDays")

    class Config:
        populate_by_name = True


class ExecutionResponse(BaseModel):
    """DCA execution response."""
    id: str
    strategy_id: str = Field(..., alias="strategyId")
    execution_number: int = Field(..., alias="executionNumber")
    status: str
    skip_reason: Optional[str] = Field(None, alias="skipReason")
    tx_hash: Optional[str] = Field(None, alias="txHash")
    actual_input_amount: Optional[str] = Field(None, alias="actualInputAmount")
    actual_output_amount: Optional[str] = Field(None, alias="actualOutputAmount")
    actual_price_usd: Optional[float] = Field(None, alias="actualPriceUsd")
    gas_usd: Optional[float] = Field(None, alias="gasUsd")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    scheduled_at: datetime = Field(..., alias="scheduledAt")
    completed_at: Optional[datetime] = Field(None, alias="completedAt")

    class Config:
        populate_by_name = True


class InternalExecuteRequest(BaseModel):
    """Request body for internal DCA execution (called by Convex cron)."""
    strategy_id: str = Field(..., alias="strategyId", description="DCA strategy ID")

    class Config:
        populate_by_name = True


# =============================================================================
# Dependencies
# =============================================================================


def get_dca_service() -> DCAService:
    """Get DCA service instance with all providers wired up."""
    convex = get_convex_client()

    from app.core.strategies.dca.providers import (
        DCASwapProvider,
        DCAPricingProvider,
        DCAGasProvider,
        DCASessionManager,
        DCAPolicyEngine,
    )

    return DCAService(
        convex_client=convex,
        swap_provider=DCASwapProvider(),
        pricing_provider=DCAPricingProvider(),
        gas_provider=DCAGasProvider(),
        session_manager=DCASessionManager(convex),
        policy_engine=DCAPolicyEngine(),
    )


def verify_internal_key(x_internal_key: str = Header(None, alias="X-Internal-Key")) -> bool:
    """Verify internal API key for cron-triggered endpoints."""
    if not x_internal_key or x_internal_key != settings.convex_internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")
    return True


# =============================================================================
# Helper Functions
# =============================================================================


def _strategy_to_response(strategy: DCAStrategy, service: DCAService) -> DCAStrategyResponse:
    """Convert DCAStrategy to response model."""
    return DCAStrategyResponse(
        id=strategy.id,
        userId=strategy.user_id,
        walletId=strategy.wallet_id,
        walletAddress=strategy.wallet_address,
        name=strategy.name,
        description=strategy.description,
        status=strategy.status,
        fromToken=strategy.config.from_token.to_dict(),
        toToken=strategy.config.to_token.to_dict(),
        amountPerExecutionUsd=float(strategy.config.amount_per_execution_usd),
        frequency=strategy.config.frequency,
        scheduleDescription=service.format_schedule_description(strategy.config),
        nextExecutionAt=strategy.next_execution_at,
        lastExecutionAt=strategy.last_execution_at,
        stats={
            "totalExecutions": strategy.stats.total_executions,
            "successfulExecutions": strategy.stats.successful_executions,
            "failedExecutions": strategy.stats.failed_executions,
            "skippedExecutions": strategy.stats.skipped_executions,
            "totalAmountSpentUsd": float(strategy.stats.total_amount_spent_usd),
            "totalTokensAcquired": str(strategy.stats.total_tokens_acquired),
            "averagePriceUsd": float(strategy.stats.average_price_usd) if strategy.stats.average_price_usd else None,
        },
        createdAt=strategy.created_at,
        updatedAt=strategy.updated_at,
    )


def _execution_to_response(execution: DCAExecution) -> ExecutionResponse:
    """Convert DCAExecution to response model."""
    return ExecutionResponse(
        id=execution.id,
        strategyId=execution.strategy_id,
        executionNumber=execution.execution_number,
        status=execution.status.value,
        skipReason=execution.skip_reason.value if execution.skip_reason else None,
        txHash=execution.tx_hash,
        actualInputAmount=str(execution.actual_input_amount) if execution.actual_input_amount else None,
        actualOutputAmount=str(execution.actual_output_amount) if execution.actual_output_amount else None,
        actualPriceUsd=float(execution.actual_price_usd) if execution.actual_price_usd else None,
        gasUsd=float(execution.gas_usd) if execution.gas_usd else None,
        errorMessage=execution.error_message,
        scheduledAt=execution.scheduled_at,
        completedAt=execution.completed_at,
    )


# =============================================================================
# Public Endpoints
# =============================================================================


@router.post("", response_model=DCAStrategyResponse)
async def create_strategy(
    request: CreateDCARequest,
    service: DCAService = Depends(get_dca_service),
):
    """Create a new DCA strategy (in draft status)."""
    try:
        config = DCAConfig(
            from_token=TokenInfo(
                symbol=request.from_token.symbol,
                address=request.from_token.address,
                chain_id=request.from_token.chain_id,
                decimals=request.from_token.decimals,
            ),
            to_token=TokenInfo(
                symbol=request.to_token.symbol,
                address=request.to_token.address,
                chain_id=request.to_token.chain_id,
                decimals=request.to_token.decimals,
            ),
            amount_per_execution_usd=Decimal(str(request.amount_per_execution_usd)),
            frequency=request.frequency,
            execution_hour_utc=request.execution_hour_utc,
            execution_day_of_week=request.execution_day_of_week,
            execution_day_of_month=request.execution_day_of_month,
            cron_expression=request.cron_expression,
            max_slippage_bps=request.max_slippage_bps,
            max_gas_usd=Decimal(str(request.max_gas_usd)),
            skip_if_gas_above_usd=Decimal(str(request.skip_if_gas_above_usd)) if request.skip_if_gas_above_usd else None,
            pause_if_price_above_usd=Decimal(str(request.pause_if_price_above_usd)) if request.pause_if_price_above_usd else None,
            pause_if_price_below_usd=Decimal(str(request.pause_if_price_below_usd)) if request.pause_if_price_below_usd else None,
            max_total_spend_usd=Decimal(str(request.max_total_spend_usd)) if request.max_total_spend_usd else None,
            max_executions=request.max_executions,
            end_date=request.end_date,
        )

        strategy = await service.create_strategy(
            user_id=request.user_id,
            wallet_id=request.wallet_id,
            wallet_address=request.wallet_address,
            name=request.name,
            config=config,
            description=request.description,
        )

        return _strategy_to_response(strategy, service)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}", response_model=DCAStrategyResponse)
async def get_strategy(
    strategy_id: str,
    service: DCAService = Depends(get_dca_service),
):
    """Get a DCA strategy by ID."""
    strategy = await service.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return _strategy_to_response(strategy, service)


@router.get("/user/{user_id}", response_model=List[DCAStrategyResponse])
async def list_user_strategies(
    user_id: str,
    status: Optional[DCAStatus] = None,
    service: DCAService = Depends(get_dca_service),
):
    """List all DCA strategies for a user."""
    strategies = await service.list_strategies(user_id=user_id, status=status)
    return [_strategy_to_response(s, service) for s in strategies]


@router.get("/wallet/{wallet_address}", response_model=List[DCAStrategyResponse])
async def list_wallet_strategies(
    wallet_address: str,
    status: Optional[DCAStatus] = None,
    service: DCAService = Depends(get_dca_service),
):
    """List all DCA strategies for a wallet."""
    strategies = await service.list_strategies(wallet_address=wallet_address, status=status)
    return [_strategy_to_response(s, service) for s in strategies]


@router.post("/{strategy_id}/activate", response_model=DCAStrategyResponse)
async def activate_strategy(
    strategy_id: str,
    request: ActivateRequest,
    service: DCAService = Depends(get_dca_service),
):
    """Activate a DCA strategy with a session key."""
    try:
        strategy = await service.activate(strategy_id, request.session_key_id)
        return _strategy_to_response(strategy, service)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{strategy_id}/pause", response_model=DCAStrategyResponse)
async def pause_strategy(
    strategy_id: str,
    reason: Optional[str] = None,
    service: DCAService = Depends(get_dca_service),
):
    """Pause an active DCA strategy."""
    try:
        strategy = await service.pause(strategy_id, reason)
        return _strategy_to_response(strategy, service)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{strategy_id}/resume", response_model=DCAStrategyResponse)
async def resume_strategy(
    strategy_id: str,
    service: DCAService = Depends(get_dca_service),
):
    """Resume a paused DCA strategy."""
    try:
        strategy = await service.resume(strategy_id)
        return _strategy_to_response(strategy, service)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{strategy_id}/stop", response_model=DCAStrategyResponse)
async def stop_strategy(
    strategy_id: str,
    service: DCAService = Depends(get_dca_service),
):
    """Stop/complete a DCA strategy."""
    strategy = await service.stop(strategy_id)
    return _strategy_to_response(strategy, service)


@router.patch("/{strategy_id}", response_model=DCAStrategyResponse)
async def update_strategy(
    strategy_id: str,
    request: UpdateConfigRequest,
    service: DCAService = Depends(get_dca_service),
):
    """Update DCA strategy configuration (only when paused or draft)."""
    try:
        updates = request.model_dump(exclude_unset=True, by_alias=False)
        strategy = await service.update_config(strategy_id, **updates)
        return _strategy_to_response(strategy, service)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{strategy_id}/executions", response_model=List[ExecutionResponse])
async def get_executions(
    strategy_id: str,
    limit: int = 50,
    service: DCAService = Depends(get_dca_service),
):
    """Get execution history for a strategy."""
    executions = await service.get_executions(strategy_id, limit)
    return [_execution_to_response(e) for e in executions]


@router.get("/{strategy_id}/session-requirements", response_model=SessionKeyRequirementsResponse)
async def get_session_requirements(
    strategy_id: str,
    service: DCAService = Depends(get_dca_service),
):
    """Get session key requirements for a strategy."""
    strategy = await service.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    requirements = service.get_session_key_requirements(strategy.config)

    return SessionKeyRequirementsResponse(
        permissions=requirements.permissions,
        valuePerTxUsd=float(requirements.value_per_tx_usd),
        totalValueUsd=float(requirements.total_value_usd),
        tokenAllowlist=requirements.token_allowlist,
        chainAllowlist=requirements.chain_allowlist,
        durationDays=requirements.duration_days,
    )


@router.get("/{strategy_id}/performance")
async def get_performance(
    strategy_id: str,
    service: DCAService = Depends(get_dca_service),
):
    """Get performance metrics for a strategy."""
    performance = await service.calculate_performance(strategy_id)
    if not performance:
        raise HTTPException(status_code=404, detail="Strategy not found or no executions")
    return performance


# =============================================================================
# Internal Endpoints (called by Convex cron)
# =============================================================================


# Called by: frontend/convex/scheduler.ts:checkDCAStrategies (line ~337)
# Convex sends: { strategyId: strategy._id } in JSON body
@router.post("/internal/execute")
async def internal_execute(
    request: InternalExecuteRequest,
    _: bool = Depends(verify_internal_key),
    service: DCAService = Depends(get_dca_service),
):
    """Internal endpoint called by cron to execute a DCA strategy."""
    try:
        result = await service.execute_now(request.strategy_id)
        return {
            "success": result.success,
            "status": result.status.value,
            "txHash": result.tx_hash,
            "errorMessage": result.error_message,
            "nextExecutionAt": result.next_execution_at.isoformat() if result.next_execution_at else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
