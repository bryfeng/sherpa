from fastapi import APIRouter, HTTPException
from ..types import ChatRequest, ChatResponse
from ..core.chat import run_chat

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat endpoint for conversational wallet analysis"""
    
    try:
        response = await run_chat(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
