from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from ..core.chat import _get_agent


router = APIRouter()


class ConversationSummary(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    last_activity: str
    message_count: int = 0
    archived: bool = False


class CreateConversationRequest(BaseModel):
    address: str = Field(description="Wallet address to bind the conversation to")
    title: Optional[str] = Field(default=None, description="Optional initial title")


class CreateConversationResponse(BaseModel):
    conversation_id: str
    title: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(address: str = Query(..., description="Wallet address to list conversations for")):
    agent = _get_agent()
    if not agent.context_manager:
        return []
    items = agent.context_manager.list_conversations(address)
    return items


@router.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation(req: CreateConversationRequest):
    agent = _get_agent()
    if not agent.context_manager:
        raise HTTPException(status_code=500, detail="Context manager unavailable")
    conv_id = agent.context_manager.create_conversation_id(req.address)
    # Apply title if provided
    if req.title is not None:
        agent.context_manager.update_conversation(conv_id, title=req.title)
    ctx = agent.context_manager._conversations.get(conv_id)
    return CreateConversationResponse(conversation_id=conv_id, title=(ctx.title if ctx else req.title))


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
async def update_conversation(conversation_id: str, req: UpdateConversationRequest):
    agent = _get_agent()
    if not agent.context_manager:
        raise HTTPException(status_code=500, detail="Context manager unavailable")
    updated = agent.context_manager.update_conversation(conversation_id, title=req.title, archived=req.archived)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return updated


class ConversationMessageModel(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = {}
    tokens: Optional[int] = None


class ConversationDetail(BaseModel):
    conversation_id: str
    owner_address: Optional[str] = None
    title: Optional[str] = None
    archived: bool = False
    created_at: str
    last_activity: str
    total_tokens: int
    message_count: int
    compressed_history: Optional[str] = None
    episodic_focus: Optional[Dict[str, Any]] = None
    portfolio_context: Optional[Dict[str, Any]] = None
    messages: List[ConversationMessageModel] = []


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Return full conversation details (for debugging/local tooling)."""
    agent = _get_agent()
    cm = agent.context_manager
    if not cm or conversation_id not in cm._conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    ctx = cm._conversations[conversation_id]
    msgs = [
        ConversationMessageModel(
            id=m.id,
            role=m.role,
            content=m.content,
            timestamp=m.timestamp.isoformat(),
            metadata=m.metadata or {},
            tokens=m.tokens,
        )
        for m in (ctx.messages or [])
    ]
    return ConversationDetail(
        conversation_id=ctx.conversation_id,
        owner_address=ctx.owner_address,
        title=ctx.title,
        archived=ctx.archived,
        created_at=ctx.created_at.isoformat(),
        last_activity=ctx.last_activity.isoformat(),
        total_tokens=ctx.total_tokens,
        message_count=ctx.message_count,
        compressed_history=ctx.compressed_history,
        episodic_focus=ctx.episodic_focus,
        portfolio_context=ctx.portfolio_context,
        messages=msgs,
    )
