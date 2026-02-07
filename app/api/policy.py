"""
Policy API Endpoints

Internal endpoints for evaluating transaction intents against the policy engine
and checking operational status.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.core.policy import (
    ActionContext,
    PolicyEngine,
    PolicyResult,
    RiskPolicyConfig,
    SystemPolicyConfig,
    ViolationSeverity,
)

router = APIRouter(prefix="/policy", tags=["Policy"])


# =============================================================================
# Request/Response Models
# =============================================================================


class PolicyStatusResponse(BaseModel):
    """Operational status of the policy engine."""
    operational: bool
    reason: Optional[str] = None
    emergency_stop: bool = False
    maintenance_mode: bool = False


class EvaluateRequest(BaseModel):
    """Request to evaluate a transaction intent against all policy layers."""
    session_id: str = Field(..., alias="sessionId")
    wallet_address: str = Field(..., alias="walletAddress")
    action_type: str = Field(..., alias="actionType", description="swap, bridge, transfer")
    chain_id: int = Field(..., alias="chainId")
    value_usd: float = Field(..., alias="valueUsd")
    contract_address: Optional[str] = Field(None, alias="contractAddress")
    token_in: Optional[str] = Field(None, alias="tokenIn")
    token_out: Optional[str] = Field(None, alias="tokenOut")
    slippage_percent: Optional[float] = Field(None, alias="slippagePercent")
    estimated_gas_usd: Optional[float] = Field(None, alias="estimatedGasUsd")

    class Config:
        populate_by_name = True


class ViolationResponse(BaseModel):
    """A single policy violation."""
    policy_type: str = Field(..., alias="policyType")
    policy_name: str = Field(..., alias="policyName")
    severity: str
    message: str
    suggestion: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Result of policy evaluation."""
    approved: bool
    violations: List[ViolationResponse] = []
    warnings: List[ViolationResponse] = []
    risk_score: float = Field(0.0, alias="riskScore")
    risk_level: str = Field("low", alias="riskLevel")
    requires_approval: bool = Field(False, alias="requiresApproval")
    approval_reason: Optional[str] = Field(None, alias="approvalReason")

    class Config:
        populate_by_name = True


# =============================================================================
# Auth
# =============================================================================


def verify_internal_key(
    x_internal_key: str = Header(None, alias="X-Internal-Key"),
) -> bool:
    """Verify internal API key for internal endpoints."""
    if not x_internal_key or x_internal_key != settings.convex_internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")
    return True


# =============================================================================
# Endpoints
# =============================================================================


# Called by: frontend/convex/ (system status checks)
@router.get("/internal/status", response_model=PolicyStatusResponse)
async def get_policy_status(
    _: bool = Depends(verify_internal_key),
) -> PolicyStatusResponse:
    """Check if the system is operational (no emergency stop or maintenance)."""
    system_config = SystemPolicyConfig()
    engine = PolicyEngine(system_config=system_config)

    is_operational, reason = engine.system_policy.is_operational()

    return PolicyStatusResponse(
        operational=is_operational,
        reason=reason,
        emergency_stop=system_config.emergency_stop,
        maintenance_mode=system_config.maintenance_mode,
    )


# Called by: frontend/convex/ or backend strategy executors
@router.post("/internal/evaluate", response_model=EvaluateResponse)
async def evaluate_policy(
    request: EvaluateRequest,
    _: bool = Depends(verify_internal_key),
) -> EvaluateResponse:
    """Evaluate a transaction intent against all 4 policy layers."""
    context = ActionContext(
        session_id=request.session_id,
        wallet_address=request.wallet_address,
        action_type=request.action_type,
        chain_id=request.chain_id,
        value_usd=Decimal(str(request.value_usd)),
        contract_address=request.contract_address,
        token_in=request.token_in,
        token_out=request.token_out,
        slippage_percent=request.slippage_percent,
        estimated_gas_usd=Decimal(str(request.estimated_gas_usd)) if request.estimated_gas_usd else None,
    )

    engine = PolicyEngine(
        system_config=SystemPolicyConfig(),
        risk_config=RiskPolicyConfig(),
    )

    result: PolicyResult = engine.evaluate(context)

    def _violation_to_response(v: Any) -> ViolationResponse:
        return ViolationResponse(
            policyType=v.policy_type.value,
            policyName=v.policy_name,
            severity=v.severity.value,
            message=v.message,
            suggestion=v.suggestion,
        )

    return EvaluateResponse(
        approved=result.approved,
        violations=[_violation_to_response(v) for v in result.violations],
        warnings=[_violation_to_response(v) for v in result.warnings],
        riskScore=result.risk_score,
        riskLevel=result.risk_level.value,
        requiresApproval=result.requires_approval,
        approvalReason=result.approval_reason,
    )
