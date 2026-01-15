from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..types import ChatRequest, ChatResponse
from ..core.chat import run_chat, stream_chat
from ..auth import optional_auth, TokenPayload

router = APIRouter()


def _validate_wallet_access(
    request: ChatRequest,
    auth: Optional[TokenPayload],
) -> None:
    """
    Validate that the authenticated user can access the requested wallet.

    If authenticated, users can only query their own wallet address.
    If not authenticated, the request is allowed but may have limited functionality.
    """
    if auth and request.address:
        # If authenticated, verify the requested address matches their wallet
        if request.address.lower() != auth.sub.lower():
            raise HTTPException(
                status_code=403,
                detail="You can only query your own wallet address"
            )


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    auth: Optional[TokenPayload] = Depends(optional_auth),
) -> ChatResponse:
    """Chat endpoint for conversational wallet analysis"""

    _validate_wallet_access(request, auth)

    try:
        response = await run_chat(request)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    auth: Optional[TokenPayload] = Depends(optional_auth),
) -> StreamingResponse:
    """Streaming chat endpoint compliant with Server-Sent Events."""

    _validate_wallet_access(request, auth)

    try:
        generator = stream_chat(request)
        return StreamingResponse(generator, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})
    except Exception as e:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")
