import asyncio
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class CacheEntry:
    value: Any
    expires_at: float
    

class TTLCache:
    """Simple in-memory TTL cache"""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            now = time.time()
            
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            if now > entry.expires_at:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        async with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            # Evict oldest if over max size
            while len(self._cache) > self.max_size:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def size(self) -> int:
        return len(self._cache)


# Global cache instance
cache = TTLCache()
