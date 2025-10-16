from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..types import ChatRequest, ChatResponse
from ..core.chat import run_chat, stream_chat

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat endpoint for conversational wallet analysis"""
    
    try:
        response = await run_chat(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint compliant with Server-Sent Events."""

    try:
        generator = stream_chat(request)
        return StreamingResponse(generator, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})
    except Exception as e:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")
