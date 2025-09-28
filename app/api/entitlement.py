from fastapi import APIRouter, HTTPException, Query

from ..services import EntitlementError, evaluate_entitlement
from ..types import EntitlementResponse

router = APIRouter()


@router.get("/entitlement", response_model=EntitlementResponse)
async def get_entitlement(
    address: str = Query(..., description="Wallet address to evaluate"),
    chain: str | None = Query(None, description="Optional chain override"),
) -> EntitlementResponse:
    try:
        result = await evaluate_entitlement(address=address, chain=chain)
    except EntitlementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
