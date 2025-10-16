from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="Message role: user or assistant")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(description="Chat conversation history")
    address: Optional[str] = Field(default=None, description="Optional wallet address for context")
    chain: str = Field(default="ethereum", description="Blockchain to query")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier for memory continuity")
    llm_provider: Optional[str] = Field(default=None, description="Override the default LLM provider for this conversation")
    llm_model: Optional[str] = Field(default=None, description="Override the default LLM model for this conversation")
