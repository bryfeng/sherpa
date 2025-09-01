from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    name: str = Field(description="Provider name")
    url: Optional[str] = Field(default=None, description="Source URL")
    

class ToolEnvelope(BaseModel):
    data: Any = Field(description="Tool result data")
    sources: List[Source] = Field(description="Data sources used")
    fetched_at: datetime = Field(description="When data was fetched")
    cached: bool = Field(default=False, description="Whether result was cached")
    latency_ms: Optional[int] = Field(default=None, description="Request latency in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or issues")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
