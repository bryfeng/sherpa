from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Provider(ABC):
    """Base provider interface"""
    
    name: str
    timeout_s: int = 10
    
    @abstractmethod
    async def ready(self) -> bool:
        """Check if provider is ready to serve requests"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return provider health status"""
        pass


class IndexerProvider(Provider):
    """Provider for blockchain indexing data (balances, transactions)"""
    
    @abstractmethod
    async def get_token_balances(self, address: str, chain: str) -> Dict[str, Any]:
        """Get all token balances for an address"""
        pass
    
    @abstractmethod
    async def get_native_balance(self, address: str, chain: str) -> Dict[str, Any]:
        """Get native token balance (ETH, etc.)"""
        pass


class PriceProvider(Provider):
    """Provider for token price data"""
    
    @abstractmethod
    async def get_token_prices(self, token_addresses: List[str], vs_currency: str = "usd") -> Dict[str, Any]:
        """Get current prices for multiple tokens"""
        pass
    
    @abstractmethod
    async def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """Get token metadata (symbol, name, decimals)"""
        pass
