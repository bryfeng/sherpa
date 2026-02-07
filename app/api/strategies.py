"""
Generic Strategy Execution API Endpoints

Internal endpoint for executing generic strategies via smart sessions.
Called by: frontend/convex/scheduler.ts:checkTriggers (smart session branch)
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.strategies.generic_executor import GenericStrategyExecutor
from app.db.convex_client import get_convex_client
from app.api.dca import verify_internal_key

router = APIRouter(prefix="/strategies", tags=["Strategies"])


# =============================================================================
# Request/Response Models
# =============================================================================


class StrategyExecuteRequest(BaseModel):
    """Request body for internal strategy execution (called by Convex cron)."""
    execution_id: str = Field(..., alias="executionId", description="Execution record ID")
    strategy_id: str = Field(..., alias="strategyId", description="Strategy ID")

    class Config:
        populate_by_name = True


class StrategyExecuteResponse(BaseModel):
    """Response from strategy execution."""
    success: bool
    execution_id: str = Field(..., alias="executionId")
    status: str
    error_message: Optional[str] = Field(None, alias="errorMessage")
    tx_hash: Optional[str] = Field(None, alias="txHash")

    class Config:
        populate_by_name = True


# =============================================================================
# Internal Endpoints
# =============================================================================


# Called by: frontend/convex/scheduler.ts:checkTriggers (smart session branch)
# Convex sends: { executionId, strategyId } in JSON body
@router.post("/internal/execute", response_model=StrategyExecuteResponse)
async def internal_execute(
    request: StrategyExecuteRequest,
    _: bool = Depends(verify_internal_key),
):
    """Internal endpoint called by cron to execute a generic strategy via smart session."""
    try:
        convex_client = get_convex_client()
        executor = GenericStrategyExecutor(convex_client=convex_client)

        result = await executor.prepare_execution(
            execution_id=request.execution_id,
        )

        return StrategyExecuteResponse(
            success=result.success,
            execution_id=result.execution_id,
            status=result.status.value if hasattr(result.status, 'value') else str(result.status),
            error_message=result.error_message,
            tx_hash=result.tx_hash,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
