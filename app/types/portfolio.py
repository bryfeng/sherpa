from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, Field


class TokenBalance(BaseModel):
    symbol: str = Field(description="Token symbol (e.g. ETH, USDC)")
    name: str = Field(description="Full token name")
    address: Optional[str] = Field(default=None, description="Token contract address (None for native ETH)")
    decimals: int = Field(description="Token decimal places")
    balance_wei: str = Field(description="Raw balance in smallest unit (wei)")
    balance_formatted: str = Field(description="Human readable balance")
    price_usd: Optional[Decimal] = Field(default=None, description="Price per token in USD")
    value_usd: Optional[Decimal] = Field(default=None, description="Total value in USD")
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }


class Portfolio(BaseModel):
    address: str = Field(description="Wallet address")
    chain: str = Field(description="Blockchain network")
    total_value_usd: Decimal = Field(description="Total portfolio value in USD")
    token_count: int = Field(description="Number of different tokens")
    tokens: List[TokenBalance] = Field(description="Individual token balances")
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }
