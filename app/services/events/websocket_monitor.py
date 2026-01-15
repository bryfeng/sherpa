"""
WebSocket-based blockchain monitoring using Alchemy's WebSocket API.

Alternative to webhooks - no public URL needed, connection initiated from backend.
"""

import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional, Set
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosed

from app.config import settings

logger = logging.getLogger(__name__)


class AlchemyWebSocketMonitor:
    """
    Monitor blockchain activity via Alchemy WebSockets.

    Usage:
        monitor = AlchemyWebSocketMonitor()
        monitor.on_activity(my_callback)
        await monitor.subscribe_address("0x123...")
        await monitor.start()
    """

    # WebSocket URLs per chain
    WS_URLS = {
        1: "wss://eth-mainnet.g.alchemy.com/v2/{key}",
        10: "wss://opt-mainnet.g.alchemy.com/v2/{key}",
        137: "wss://polygon-mainnet.g.alchemy.com/v2/{key}",
        42161: "wss://arb-mainnet.g.alchemy.com/v2/{key}",
        8453: "wss://base-mainnet.g.alchemy.com/v2/{key}",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.alchemy_api_key
        self._watched_addresses: Set[str] = set()
        self._connections: Dict[int, websockets.WebSocketClientProtocol] = {}
        self._subscription_ids: Dict[int, List[str]] = {}
        self._callbacks: List[Callable] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def on_activity(self, callback: Callable) -> None:
        """Register a callback for activity events."""
        self._callbacks.append(callback)

    async def subscribe_address(self, address: str, chain_id: int = 1) -> None:
        """Add an address to monitor."""
        self._watched_addresses.add(address.lower())

        # If already connected, add subscription
        if chain_id in self._connections:
            await self._add_subscription(chain_id, address)

    async def unsubscribe_address(self, address: str) -> None:
        """Remove an address from monitoring."""
        self._watched_addresses.discard(address.lower())

    async def start(self, chain_ids: Optional[List[int]] = None) -> None:
        """Start WebSocket connections for specified chains."""
        if self._running:
            return

        self._running = True
        chain_ids = chain_ids or [1]  # Default to Ethereum mainnet

        for chain_id in chain_ids:
            task = asyncio.create_task(self._run_chain_monitor(chain_id))
            self._tasks.append(task)

        logger.info(f"WebSocket monitor started for chains: {chain_ids}")

    async def stop(self) -> None:
        """Stop all WebSocket connections."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        for ws in self._connections.values():
            await ws.close()

        self._connections.clear()
        self._subscription_ids.clear()
        self._tasks.clear()

        logger.info("WebSocket monitor stopped")

    async def _run_chain_monitor(self, chain_id: int) -> None:
        """Run WebSocket connection for a single chain with auto-reconnect."""
        ws_url = self.WS_URLS.get(chain_id)
        if not ws_url:
            logger.error(f"No WebSocket URL for chain {chain_id}")
            return

        url = ws_url.format(key=self.api_key)
        retry_delay = 1
        max_retry_delay = 60

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._connections[chain_id] = ws
                    self._subscription_ids[chain_id] = []
                    retry_delay = 1  # Reset on successful connection

                    logger.info(f"Connected to Alchemy WebSocket for chain {chain_id}")

                    # Subscribe to addresses
                    for address in self._watched_addresses:
                        await self._add_subscription(chain_id, address)

                    # Listen for messages
                    await self._listen(chain_id, ws)

            except ConnectionClosed as e:
                logger.warning(f"WebSocket closed for chain {chain_id}: {e}")
            except Exception as e:
                logger.error(f"WebSocket error for chain {chain_id}: {e}")

            if self._running:
                logger.info(f"Reconnecting to chain {chain_id} in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    async def _add_subscription(self, chain_id: int, address: str) -> None:
        """Add a subscription for an address."""
        ws = self._connections.get(chain_id)
        if not ws:
            return

        # Subscribe to logs (ERC20 transfers) for this address
        # This catches both incoming and outgoing transfers
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": [
                "logs",
                {
                    "address": None,  # Any contract
                    "topics": [
                        # Transfer(address,address,uint256) - ERC20 Transfer event
                        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                        None,  # from (any)
                        None,  # to (any)
                    ]
                }
            ]
        }

        # Also subscribe to alchemy_minedTransactions for this address
        mined_tx_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_subscribe",
            "params": [
                "alchemy_minedTransactions",
                {
                    "addresses": [{"from": address}, {"to": address}],
                    "includeRemoved": False,
                    "hashesOnly": False,
                }
            ]
        }

        try:
            await ws.send(json.dumps(mined_tx_msg))
            response = await ws.recv()
            result = json.loads(response)

            if "result" in result:
                self._subscription_ids[chain_id].append(result["result"])
                logger.info(f"Subscribed to mined transactions for {address} on chain {chain_id}")
            elif "error" in result:
                logger.error(f"Subscription error: {result['error']}")

        except Exception as e:
            logger.error(f"Failed to subscribe for {address}: {e}")

    async def _listen(self, chain_id: int, ws) -> None:
        """Listen for WebSocket messages."""
        async for message in ws:
            try:
                data = json.loads(message)

                if "method" in data and data["method"] == "eth_subscription":
                    await self._handle_subscription_event(chain_id, data)

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {message[:100]}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")

    async def _handle_subscription_event(self, chain_id: int, data: dict) -> None:
        """Handle a subscription event."""
        params = data.get("params", {})
        result = params.get("result", {})

        # Extract transaction info
        tx_hash = result.get("hash") or result.get("transactionHash")
        from_addr = result.get("from", "").lower()
        to_addr = result.get("to", "").lower()

        # Check if this involves a watched address
        involved_address = None
        direction = "internal"

        if from_addr in self._watched_addresses:
            involved_address = from_addr
            direction = "out"
        elif to_addr in self._watched_addresses:
            involved_address = to_addr
            direction = "in"

        if not involved_address:
            return

        # Build activity event
        activity = {
            "wallet_address": involved_address,
            "chain_id": chain_id,
            "tx_hash": tx_hash,
            "from_address": from_addr,
            "to_address": to_addr,
            "direction": direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw": result,
        }

        logger.info(f"Activity detected: {direction} transaction {tx_hash[:16]}... for {involved_address[:10]}...")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(activity)
                else:
                    callback(activity)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# Singleton instance
_monitor: Optional[AlchemyWebSocketMonitor] = None


def get_websocket_monitor() -> AlchemyWebSocketMonitor:
    """Get the singleton WebSocket monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = AlchemyWebSocketMonitor()
    return _monitor
