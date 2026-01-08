from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.chat import _get_agent


router = APIRouter()


class ConversationSummary(BaseModel):
    conversation_id: str
    convex_id: Optional[str] = None  # Convex ID for persistence
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
    """List conversations for a wallet address.

    Combines in-memory cache with Convex persisted conversations.
    """
    agent = _get_agent()
    if not agent.context_manager:
        return []

    # Get in-memory conversations
    in_memory = agent.context_manager.list_conversations(address)
    in_memory_ids = {item.get('conversation_id') for item in in_memory}

    # Also fetch from Convex for persisted conversations
    convex_conversations = []
    if agent.context_manager.convex_client:
        try:
            convex_conversations = await agent.context_manager.load_conversations_from_convex(address)
        except Exception as e:
            # Log but don't fail - in-memory data still available
            pass

    # Merge results, preferring in-memory for recent activity
    result = []
    seen_convex_ids = set()

    # Add in-memory items first
    for item in in_memory:
        ctx = agent.context_manager._conversations.get(item.get('conversation_id'))
        result.append(ConversationSummary(
            conversation_id=item.get('conversation_id', ''),
            convex_id=ctx.convex_id if ctx else None,
            title=item.get('title'),
            last_activity=item.get('last_activity', datetime.now().isoformat()),
            message_count=item.get('message_count', 0),
            archived=item.get('archived', False),
        ))
        if ctx and ctx.convex_id:
            seen_convex_ids.add(ctx.convex_id)

    # Add Convex conversations not in memory
    for conv in convex_conversations:
        convex_id = conv.get('_id')
        if convex_id and convex_id not in seen_convex_ids:
            result.append(ConversationSummary(
                conversation_id=convex_id,  # Use Convex ID as conversation_id
                convex_id=convex_id,
                title=conv.get('title'),
                last_activity=datetime.fromtimestamp(conv.get('updatedAt', 0) / 1000).isoformat() if conv.get('updatedAt') else datetime.now().isoformat(),
                message_count=conv.get('totalTokens', 0),  # Approximate
                archived=conv.get('archived', False),
            ))

    # Sort by last_activity descending
    result.sort(key=lambda x: x.last_activity, reverse=True)
    return result


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
    """Return full conversation details.

    First checks in-memory cache, then falls back to Convex if not found.
    """
    agent = _get_agent()
    cm = agent.context_manager
    if not cm:
        raise HTTPException(status_code=500, detail="Context manager unavailable")

    # Check in-memory first
    ctx = cm._conversations.get(conversation_id)
    if ctx:
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

    # Try loading from Convex (conversation_id might be a Convex ID)
    if cm.convex_client:
        try:
            convex_data = await cm.load_conversation_messages(conversation_id)
            if convex_data:
                msgs = [
                    ConversationMessageModel(
                        id=m.get('_id', ''),
                        role=m.get('role', 'user'),
                        content=m.get('content', ''),
                        timestamp=datetime.fromtimestamp(m.get('createdAt', 0) / 1000).isoformat() if m.get('createdAt') else datetime.now().isoformat(),
                        metadata=m.get('metadata') or {},
                        tokens=m.get('tokenCount'),
                    )
                    for m in (convex_data.get('messages') or [])
                ]
                return ConversationDetail(
                    conversation_id=conversation_id,
                    owner_address=None,  # Not stored in Convex directly
                    title=convex_data.get('title'),
                    archived=convex_data.get('archived', False),
                    created_at=datetime.fromtimestamp(convex_data.get('createdAt', 0) / 1000).isoformat() if convex_data.get('createdAt') else datetime.now().isoformat(),
                    last_activity=datetime.fromtimestamp(convex_data.get('updatedAt', 0) / 1000).isoformat() if convex_data.get('updatedAt') else datetime.now().isoformat(),
                    total_tokens=convex_data.get('totalTokens', 0),
                    message_count=len(msgs),
                    compressed_history=None,
                    episodic_focus=None,
                    portfolio_context=None,
                    messages=msgs,
                )
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Conversation not found")
