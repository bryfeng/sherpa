from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..providers.relay import RelayProvider

router = APIRouter(prefix="/tools/relay")


class RelayQuoteRequest(BaseModel):
    user: str = Field(..., description="EOA initiating the bridge")
    originChainId: int = Field(..., description="Source chain ID")
    destinationChainId: int = Field(..., description="Destination chain ID")
    originCurrency: str = Field(..., description="Input token address or zero-address for native")
    destinationCurrency: str = Field(..., description="Output token address or zero-address for native")
    recipient: str = Field(..., description="Recipient address on the destination chain")
    tradeType: str = Field('EXACT_INPUT', description="Relay trade type", pattern='^(EXACT_INPUT|EXACT_OUTPUT)$')
    amount: str = Field(..., description="Amount in smallest units (wei / token decimals)")
    referrer: Optional[str] = Field(default='sherpa.chat', description="Attribution referrer")
    useExternalLiquidity: bool = False
    useDepositAddress: bool = False
    topupGas: bool = False
    usePermit: Optional[bool] = None
    slippageTolerance: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        # Relay expects lowercase keys; pydantic already produces them, just ensure user key name matches API.
        payload['user'] = payload.pop('user')
        return payload


@router.post("/quote")
async def relay_quote(request: RelayQuoteRequest) -> Dict[str, Any]:
    provider = RelayProvider()
    try:
        data = await provider.quote(request.to_payload())
        return {'success': True, 'quote': data}
    except httpx.HTTPStatusError as exc:  # type: ignore[assignment]
        detail = exc.response.text or exc.response.reason_phrase
        raise HTTPException(status_code=exc.response.status_code, detail=detail)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f'Failed to fetch Relay quote: {exc}')


@router.get("/requests/{request_id}/signature")
async def relay_signature(request_id: str) -> Dict[str, Any]:
    provider = RelayProvider()
    try:
        data = await provider.get_request_signature(request_id)
        return {'success': True, 'signature': data}
    except httpx.HTTPStatusError as exc:  # type: ignore[assignment]
        detail = exc.response.text or exc.response.reason_phrase
        raise HTTPException(status_code=exc.response.status_code, detail=detail)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f'Failed to fetch Relay signature: {exc}')
