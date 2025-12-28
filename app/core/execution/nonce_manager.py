"""
Nonce management for concurrent transactions.

Handles nonce tracking to prevent conflicts when multiple transactions
are being prepared or executed concurrently.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
import httpx

from app.config import settings


@dataclass
class NonceState:
    """Tracks nonce state for an address on a chain."""
    address: str
    chain_id: int
    confirmed_nonce: int                        # Last confirmed on-chain
    pending_nonce: int                          # Next available for use
    reserved_nonces: Set[int] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class NonceManager:
    """
    Manages nonces for concurrent transaction execution.

    Features:
    - Tracks pending nonces to avoid conflicts
    - Auto-syncs with on-chain state
    - Handles nonce gaps and releases
    - Thread-safe for concurrent access
    """

    def __init__(self, rpc_urls: Optional[Dict[int, str]] = None):
        self._states: Dict[str, NonceState] = {}  # key: "{chain_id}:{address}"
        self._locks: Dict[str, asyncio.Lock] = {}
        self._client = httpx.AsyncClient(timeout=30.0)

        # Default RPC URLs (can be overridden)
        self._rpc_urls = rpc_urls or {
            1: f"https://eth-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            10: f"https://opt-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            137: f"https://polygon-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            42161: f"https://arb-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            8453: f"https://base-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
        }

    def _get_key(self, chain_id: int, address: str) -> str:
        return f"{chain_id}:{address.lower()}"

    async def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def _fetch_on_chain_nonce(self, chain_id: int, address: str) -> int:
        """Fetch the current nonce from the blockchain."""
        rpc_url = self._rpc_urls.get(chain_id)
        if not rpc_url:
            raise ValueError(f"No RPC URL configured for chain {chain_id}")

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getTransactionCount",
            "params": [address, "pending"],
            "id": 1,
        }

        response = await self._client.post(rpc_url, json=payload)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise RuntimeError(f"RPC error: {result['error']}")

        return int(result["result"], 16)

    async def get_next_nonce(
        self,
        address: str,
        chain_id: int,
        sync: bool = True,
    ) -> int:
        """
        Get the next available nonce for an address.

        Args:
            address: The wallet address
            chain_id: The chain ID
            sync: Whether to sync with on-chain state first

        Returns:
            The next available nonce (already reserved)
        """
        key = self._get_key(chain_id, address)
        lock = await self._get_lock(key)

        async with lock:
            # Initialize or refresh state if needed
            if key not in self._states or sync:
                on_chain_nonce = await self._fetch_on_chain_nonce(chain_id, address)

                if key not in self._states:
                    self._states[key] = NonceState(
                        address=address.lower(),
                        chain_id=chain_id,
                        confirmed_nonce=on_chain_nonce,
                        pending_nonce=on_chain_nonce,
                    )
                else:
                    # Update confirmed nonce, but don't decrease pending
                    state = self._states[key]
                    state.confirmed_nonce = on_chain_nonce
                    if on_chain_nonce > state.pending_nonce:
                        state.pending_nonce = on_chain_nonce
                    state.last_updated = datetime.utcnow()

            state = self._states[key]

            # Find the next available nonce
            nonce = state.pending_nonce
            while nonce in state.reserved_nonces:
                nonce += 1

            # Reserve it
            state.reserved_nonces.add(nonce)
            state.pending_nonce = nonce + 1

            return nonce

    async def release_nonce(
        self,
        address: str,
        chain_id: int,
        nonce: int,
    ) -> None:
        """
        Release a reserved nonce (e.g., transaction failed before broadcast).

        Args:
            address: The wallet address
            chain_id: The chain ID
            nonce: The nonce to release
        """
        key = self._get_key(chain_id, address)
        lock = await self._get_lock(key)

        async with lock:
            if key in self._states:
                state = self._states[key]
                state.reserved_nonces.discard(nonce)

                # If we released the highest nonce, we can reduce pending
                if nonce == state.pending_nonce - 1:
                    while state.pending_nonce > state.confirmed_nonce:
                        if state.pending_nonce - 1 not in state.reserved_nonces:
                            state.pending_nonce -= 1
                        else:
                            break

    async def confirm_nonce(
        self,
        address: str,
        chain_id: int,
        nonce: int,
    ) -> None:
        """
        Mark a nonce as confirmed (transaction included in block).

        Args:
            address: The wallet address
            chain_id: The chain ID
            nonce: The confirmed nonce
        """
        key = self._get_key(chain_id, address)
        lock = await self._get_lock(key)

        async with lock:
            if key in self._states:
                state = self._states[key]
                state.reserved_nonces.discard(nonce)

                # Update confirmed nonce if this is higher
                if nonce >= state.confirmed_nonce:
                    state.confirmed_nonce = nonce + 1

    async def sync_with_chain(
        self,
        address: str,
        chain_id: int,
    ) -> int:
        """
        Sync nonce state with on-chain data.

        Returns the current on-chain pending nonce.
        """
        key = self._get_key(chain_id, address)
        lock = await self._get_lock(key)

        async with lock:
            on_chain_nonce = await self._fetch_on_chain_nonce(chain_id, address)

            if key in self._states:
                state = self._states[key]
                state.confirmed_nonce = on_chain_nonce

                # Clear any reserved nonces that are now confirmed
                state.reserved_nonces = {
                    n for n in state.reserved_nonces if n >= on_chain_nonce
                }

                # Update pending if on-chain is higher
                if on_chain_nonce > state.pending_nonce:
                    state.pending_nonce = on_chain_nonce

                state.last_updated = datetime.utcnow()
            else:
                self._states[key] = NonceState(
                    address=address.lower(),
                    chain_id=chain_id,
                    confirmed_nonce=on_chain_nonce,
                    pending_nonce=on_chain_nonce,
                )

            return on_chain_nonce

    async def get_state(
        self,
        address: str,
        chain_id: int,
    ) -> Optional[NonceState]:
        """Get the current nonce state for an address."""
        key = self._get_key(chain_id, address)
        return self._states.get(key)

    async def clear_state(
        self,
        address: str,
        chain_id: int,
    ) -> None:
        """Clear cached nonce state for an address."""
        key = self._get_key(chain_id, address)
        self._states.pop(key, None)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# Singleton instance
_nonce_manager: Optional[NonceManager] = None


def get_nonce_manager() -> NonceManager:
    """Get the singleton nonce manager instance."""
    global _nonce_manager
    if _nonce_manager is None:
        _nonce_manager = NonceManager()
    return _nonce_manager
