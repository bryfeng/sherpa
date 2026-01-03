"""
Event Processor

Classifies, enriches, and routes events to subscribers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

from .models import (
    ChainType,
    EventType,
    WalletActivity,
    ParsedTransaction,
    TokenTransfer,
    TransactionAction,
    SwapDetails,
    EventCallback,
)

logger = logging.getLogger(__name__)


# Known DEX router addresses for swap classification
DEX_ROUTERS: Dict[str, Dict[str, str]] = {
    # Ethereum
    "ethereum": {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3",
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "uniswap_v2",
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3_router",
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "uniswap_universal",
        "0xdef1c0ded9bec7f1a1670819833240f027b25eff": "0x_exchange",
        "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch_v4",
        "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch_v5",
        "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "sushiswap",
    },
    # Arbitrum
    "arbitrum": {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3",
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3_router",
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": "sushiswap",
        "0xc873fecbd354f5a56e00e710b90ef4201db2448d": "camelot",
    },
    # Polygon
    "polygon": {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3",
        "0xa5e0829caced8ffdd4de3c43696c57f7d7a678ff": "quickswap",
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": "sushiswap",
    },
    # Base
    "base": {
        "0x2626664c2603336e57b271c5c0b26f421741e481": "uniswap_v3",
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "uniswap_universal",
        "0x6bded42c6da8fbf0d2ba55b2fa120c5e0c8d7891": "aerodrome",
    },
}

# Known DeFi protocol addresses
DEFI_PROTOCOLS: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "aave_v2",
        "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "aave_v3",
        "0xc3d688b66703497daa19211eedff47f25384cdc3": "compound_v3",
        "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "lido",
        "0xcc9ee9483f662091a1de4795249e24ac0ac2630f": "rocketpool",
    },
    "arbitrum": {
        "0x794a61358d6845594f94dc1db02a252b5b4814ad": "aave_v3",
        "0xa97684ead0e402dc232d5a977953df7ecbab3cdb": "aave_v3_pool",
    },
}

# Bridge contracts
BRIDGE_CONTRACTS: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "0x3154cf16ccdb4c6d922629664174b904d80f2c35": "relay",
        "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1": "optimism_gateway",
        "0x4dbd4fc535ac27206064b68ffcf827b0a60bab3f": "arbitrum_inbox",
        "0x3ee18b2214aff97000d974cf647e7c347e8fa585": "wormhole",
    },
}


class EventProcessor:
    """
    Process and enrich wallet activities.

    Responsibilities:
    - Classify transaction types (swap, bridge, stake, etc.)
    - Enrich with protocol information
    - Calculate USD values
    - Route to appropriate handlers
    - Deduplicate events
    """

    def __init__(
        self,
        price_provider: Optional[Any] = None,
        convex_client: Optional[Any] = None,
    ):
        self.price_provider = price_provider
        self.convex_client = convex_client

        # Callbacks for different event types
        self._callbacks: Dict[EventType, List[EventCallback]] = {}

        # Seen events for deduplication (tx_hash -> timestamp)
        self._seen_events: Dict[str, datetime] = {}
        self._seen_ttl_seconds = 300  # 5 minutes

    def register_callback(
        self,
        event_type: EventType,
        callback: EventCallback,
    ):
        """Register a callback for an event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def unregister_callback(
        self,
        event_type: EventType,
        callback: EventCallback,
    ):
        """Unregister a callback."""
        if event_type in self._callbacks:
            self._callbacks[event_type] = [
                cb for cb in self._callbacks[event_type] if cb != callback
            ]

    async def process_activities(
        self,
        activities: List[WalletActivity],
    ) -> List[WalletActivity]:
        """
        Process a batch of wallet activities.

        - Deduplicates
        - Classifies and enriches
        - Routes to callbacks
        - Persists to storage
        """
        processed: List[WalletActivity] = []

        for activity in activities:
            # Deduplicate
            if self._is_duplicate(activity):
                logger.debug(f"Skipping duplicate event: {activity.tx_hash}")
                continue

            # Classify and enrich
            enriched = await self._enrich_activity(activity)

            # Mark as seen
            self._mark_seen(enriched)

            # Route to callbacks
            await self._route_to_callbacks(enriched)

            # Persist
            await self._persist_activity(enriched)

            processed.append(enriched)

        # Cleanup old seen events
        self._cleanup_seen_events()

        return processed

    def _is_duplicate(self, activity: WalletActivity) -> bool:
        """Check if we've already processed this event."""
        key = f"{activity.tx_hash}:{activity.wallet_address}:{activity.event_type}"
        return key in self._seen_events

    def _mark_seen(self, activity: WalletActivity):
        """Mark an event as seen."""
        key = f"{activity.tx_hash}:{activity.wallet_address}:{activity.event_type}"
        self._seen_events[key] = datetime.now(timezone.utc)

    def _cleanup_seen_events(self):
        """Remove old entries from seen events."""
        now = datetime.now(timezone.utc)
        expired = [
            key for key, ts in self._seen_events.items()
            if (now - ts).total_seconds() > self._seen_ttl_seconds
        ]
        for key in expired:
            del self._seen_events[key]

    async def _enrich_activity(
        self,
        activity: WalletActivity,
    ) -> WalletActivity:
        """Enrich activity with additional classification and data."""
        # Reclassify based on deeper analysis
        activity = self._classify_event_type(activity)

        # Add protocol labels
        activity = self._add_protocol_labels(activity)

        # Calculate USD values if possible
        activity = await self._calculate_usd_values(activity)

        # Determine if copyable
        activity = self._assess_copyability(activity)

        # Mark as processed
        activity.processed_at = datetime.now(timezone.utc)

        return activity

    def _classify_event_type(
        self,
        activity: WalletActivity,
    ) -> WalletActivity:
        """Reclassify event type based on contract interactions."""
        chain = activity.chain.value
        counterparty = (activity.counterparty_address or "").lower()

        # Check if it's a DEX interaction (swap)
        dex_routers = DEX_ROUTERS.get(chain, {})
        if counterparty in dex_routers:
            activity.event_type = EventType.SWAP
            activity.counterparty_label = dex_routers[counterparty]
            return activity

        # Check if it's a DeFi protocol
        defi_protocols = DEFI_PROTOCOLS.get(chain, {})
        if counterparty in defi_protocols:
            protocol = defi_protocols[counterparty]
            activity.counterparty_label = protocol

            # Determine action based on direction
            if activity.direction == "out":
                if "aave" in protocol or "compound" in protocol:
                    activity.event_type = EventType.DEPOSIT
                elif "lido" in protocol or "rocket" in protocol:
                    activity.event_type = EventType.STAKE
            else:
                if "aave" in protocol or "compound" in protocol:
                    activity.event_type = EventType.WITHDRAW
            return activity

        # Check if it's a bridge
        bridge_contracts = BRIDGE_CONTRACTS.get(chain, {})
        if counterparty in bridge_contracts:
            activity.event_type = EventType.BRIDGE_INITIATE
            activity.counterparty_label = bridge_contracts[counterparty]
            return activity

        return activity

    def _add_protocol_labels(
        self,
        activity: WalletActivity,
    ) -> WalletActivity:
        """Add protocol labels to parsed transaction."""
        if not activity.parsed_tx:
            return activity

        chain = activity.chain.value
        contract = (activity.parsed_tx.to_address or "").lower()

        # Check all known contracts
        for contracts in [DEX_ROUTERS, DEFI_PROTOCOLS, BRIDGE_CONTRACTS]:
            chain_contracts = contracts.get(chain, {})
            if contract in chain_contracts:
                if activity.parsed_tx.actions:
                    activity.parsed_tx.actions[0].protocol = chain_contracts[contract]
                break

        return activity

    async def _calculate_usd_values(
        self,
        activity: WalletActivity,
    ) -> WalletActivity:
        """Calculate USD values for transfers."""
        if not self.price_provider:
            return activity

        if activity.parsed_tx and activity.parsed_tx.token_transfers:
            for transfer in activity.parsed_tx.token_transfers:
                if transfer.amount and not transfer.value_usd:
                    try:
                        price = await self.price_provider.get_token_price(
                            transfer.token_address,
                            activity.chain.value,
                        )
                        if price:
                            transfer.value_usd = transfer.amount * Decimal(str(price))
                    except Exception as e:
                        logger.debug(f"Could not get price for {transfer.token_address}: {e}")

        return activity

    def _assess_copyability(
        self,
        activity: WalletActivity,
    ) -> WalletActivity:
        """Determine if this activity is suitable for copy trading."""
        # Copyable events
        copyable_types = {
            EventType.SWAP,
            EventType.BRIDGE_INITIATE,
            EventType.DEPOSIT,
            EventType.STAKE,
        }

        if activity.event_type in copyable_types:
            activity.is_copyable = True

            # Calculate relevance score (0-1)
            score = 0.5  # Base score

            # Higher score for swaps
            if activity.event_type == EventType.SWAP:
                score += 0.2

            # Higher score if we have USD value
            if activity.value_usd:
                score += 0.1
                # Higher score for larger trades
                if activity.value_usd > Decimal("1000"):
                    score += 0.1
                if activity.value_usd > Decimal("10000"):
                    score += 0.1

            activity.copy_relevance_score = min(score, 1.0)
        else:
            activity.is_copyable = False
            activity.copy_relevance_score = 0.0

        return activity

    async def _route_to_callbacks(
        self,
        activity: WalletActivity,
    ):
        """Route activity to registered callbacks."""
        # Get callbacks for this specific event type
        callbacks = self._callbacks.get(activity.event_type, [])

        # Also get callbacks registered for "all" (if we add that)
        # callbacks.extend(self._callbacks.get(EventType.ALL, []))

        if not callbacks:
            return

        # Execute callbacks concurrently
        tasks = [cb(activity) for cb in callbacks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Callback error: {result}")

    async def _persist_activity(
        self,
        activity: WalletActivity,
    ):
        """Persist activity to Convex."""
        if not self.convex_client:
            return

        try:
            await self.convex_client.mutation(
                "walletActivity:insert",
                {
                    "id": activity.id,
                    "walletAddress": activity.wallet_address,
                    "chain": activity.chain.value,
                    "eventType": activity.event_type.value,
                    "txHash": activity.tx_hash,
                    "blockNumber": activity.block_number,
                    "timestamp": int(activity.timestamp.timestamp() * 1000),
                    "direction": activity.direction,
                    "valueUsd": float(activity.value_usd) if activity.value_usd else None,
                    "counterpartyAddress": activity.counterparty_address,
                    "counterpartyLabel": activity.counterparty_label,
                    "isCopyable": activity.is_copyable,
                    "copyRelevanceScore": activity.copy_relevance_score,
                    "createdAt": int(activity.created_at.timestamp() * 1000),
                    "processedAt": int(activity.processed_at.timestamp() * 1000) if activity.processed_at else None,
                },
            )
        except Exception as e:
            logger.error(f"Failed to persist activity: {e}")


class CopyTradingEventFilter:
    """
    Filter events for copy trading relevance.

    Determines which events should trigger copy trades.
    """

    def __init__(
        self,
        min_value_usd: Decimal = Decimal("10"),
        max_value_usd: Optional[Decimal] = None,
        allowed_types: Optional[Set[EventType]] = None,
        blocked_tokens: Optional[Set[str]] = None,
    ):
        self.min_value_usd = min_value_usd
        self.max_value_usd = max_value_usd
        self.allowed_types = allowed_types or {EventType.SWAP}
        self.blocked_tokens = blocked_tokens or set()

    def should_copy(self, activity: WalletActivity) -> bool:
        """Determine if this activity should be copied."""
        # Must be copyable
        if not activity.is_copyable:
            return False

        # Must be allowed type
        if activity.event_type not in self.allowed_types:
            return False

        # Check value bounds
        if activity.value_usd:
            if activity.value_usd < self.min_value_usd:
                return False
            if self.max_value_usd and activity.value_usd > self.max_value_usd:
                return False

        # Check blocked tokens
        if activity.parsed_tx:
            for transfer in activity.parsed_tx.token_transfers:
                if transfer.token_address.lower() in self.blocked_tokens:
                    return False

        return True

    def get_trade_details(self, activity: WalletActivity) -> Optional[Dict[str, Any]]:
        """Extract trade details from a copyable activity."""
        if not self.should_copy(activity):
            return None

        if activity.event_type == EventType.SWAP and activity.parsed_tx:
            # Find input and output tokens
            transfers = activity.parsed_tx.token_transfers
            if len(transfers) >= 2:
                # Assume first is input, last is output (simplified)
                token_in = transfers[0]
                token_out = transfers[-1]

                return {
                    "type": "swap",
                    "token_in": {
                        "address": token_in.token_address,
                        "symbol": token_in.token_symbol,
                        "amount": str(token_in.amount) if token_in.amount else token_in.amount_raw,
                    },
                    "token_out": {
                        "address": token_out.token_address,
                        "symbol": token_out.token_symbol,
                        "amount": str(token_out.amount) if token_out.amount else token_out.amount_raw,
                    },
                    "value_usd": float(activity.value_usd) if activity.value_usd else None,
                    "dex": activity.counterparty_label,
                }

        return None
