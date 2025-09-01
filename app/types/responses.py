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
