from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from .portfolio import Portfolio


class PortfolioResponse(BaseModel):
    success: bool = Field(description="Whether request was successful")
    portfolio: Optional[Portfolio] = Field(default=None, description="Portfolio data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    sources: list = Field(default_factory=list, description="Data sources used")


class ChatResponse(BaseModel):
    reply: str = Field(description="Chat response text")
    panels: Dict[str, Any] = Field(default_factory=dict, description="Structured data panels")
    sources: list = Field(default_factory=list, description="Data sources used")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier for memory continuity")
    llm_provider: Optional[str] = Field(default=None, description="Canonical identifier of the LLM provider that generated the response")
    llm_model: Optional[str] = Field(default=None, description="Identifier of the LLM model used for the response")


class EntitlementResponse(BaseModel):
    address: str = Field(description="Wallet address that was evaluated")
    chain: str = Field(description="Chain used for entitlement evaluation")
    pro: bool = Field(description="Whether the wallet currently has Pro access")
    gating: str = Field(description="Entitlement mechanism (token, disabled, error)")
    standard: Optional[str] = Field(default=None, description="Token standard if gating is token-based")
    token_address: Optional[str] = Field(default=None, description="Entitlement token contract address")
    token_id: Optional[str] = Field(default=None, description="Specific token ID for ERC-1155 gating")
    reason: Optional[str] = Field(default=None, description="Explanation when Pro access is unavailable")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of evaluation")
    cached: bool = Field(default=False, description="Indicates whether the result was served from cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional entitlement metadata")
