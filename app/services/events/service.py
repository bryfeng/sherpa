"""
Event Monitoring Service

Main service that orchestrates webhook handlers, subscriptions, and event processing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Dict, List, Optional

from .models import (
    ChainType,
    EventType,
    Subscription,
    SubscriptionStatus,
    WalletActivity,
    WebhookPayload,
    WebhookConfig,
    WebhookRegistration,
    EventCallback,
)
from .webhook_handler import AlchemyWebhookHandler, HeliusWebhookHandler, WebhookHandler
from .event_processor import EventProcessor

logger = logging.getLogger(__name__)


# Singleton instance
_service_instance: Optional["EventMonitoringService"] = None


def get_event_monitoring_service() -> "EventMonitoringService":
    """Get the singleton EventMonitoringService instance."""
    global _service_instance
    if _service_instance is None:
        from ...config import settings
        from ...db import get_convex_client

        _service_instance = EventMonitoringService(
            alchemy_api_key=settings.alchemy_api_key,
            alchemy_signing_key=getattr(settings, "alchemy_webhook_signing_key", None),
            helius_api_key=getattr(settings, "helius_api_key", None),
            convex_client=get_convex_client(),
            webhook_base_url=getattr(settings, "webhook_base_url", None),
        )

    return _service_instance


class EventMonitoringService:
    """
    Unified event monitoring service.

    Manages:
    - Webhook handlers for different chains
    - Address subscriptions
    - Event processing and routing
    - Callback registration
    """

    def __init__(
        self,
        alchemy_api_key: Optional[str] = None,
        alchemy_signing_key: Optional[str] = None,
        helius_api_key: Optional[str] = None,
        convex_client: Optional[Any] = None,
        webhook_base_url: Optional[str] = None,
        price_provider: Optional[Any] = None,
    ):
        self.convex_client = convex_client
        self.webhook_base_url = webhook_base_url

        # Initialize handlers
        self._handlers: Dict[str, WebhookHandler] = {}

        if alchemy_api_key:
            self._handlers["alchemy"] = AlchemyWebhookHandler(
                api_key=alchemy_api_key,
                signing_key=alchemy_signing_key,
            )

        if helius_api_key:
            self._handlers["helius"] = HeliusWebhookHandler(
                api_key=helius_api_key,
            )

        # Initialize processor
        self._processor = EventProcessor(
            price_provider=price_provider,
            convex_client=convex_client,
        )

        # In-memory subscription cache
        self._subscriptions: Dict[str, Subscription] = {}
        self._address_to_subscription: Dict[str, str] = {}  # address -> subscription_id

        # Webhook registrations by provider
        self._webhook_registrations: Dict[str, WebhookRegistration] = {}

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe_address(
        self,
        address: str,
        chain: ChainType,
        user_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        label: Optional[str] = None,
    ) -> Subscription:
        """
        Subscribe to events for an address.

        Creates a subscription and registers with the appropriate webhook provider.
        """
        address_lower = address.lower()

        # Check if already subscribed
        existing_id = self._address_to_subscription.get(f"{chain.value}:{address_lower}")
        if existing_id and existing_id in self._subscriptions:
            existing = self._subscriptions[existing_id]
            logger.info(f"Address {address} already subscribed on {chain.value}")
            return existing

        # Create subscription
        subscription = Subscription(
            user_id=user_id,
            address=address_lower,
            chain=chain,
            event_types=event_types or list(EventType),
            label=label,
            status=SubscriptionStatus.PENDING,
        )

        # Register webhook with provider
        try:
            webhook_reg = await self._register_webhook_for_address(
                address=address_lower,
                chain=chain,
            )
            subscription.webhook_id = webhook_reg.webhook_id
            subscription.status = SubscriptionStatus.ACTIVE
        except Exception as e:
            logger.error(f"Failed to register webhook for {address}: {e}")
            subscription.status = SubscriptionStatus.FAILED
            subscription.error_message = str(e)

        # Store subscription
        self._subscriptions[subscription.id] = subscription
        self._address_to_subscription[f"{chain.value}:{address_lower}"] = subscription.id

        # Persist to Convex
        await self._persist_subscription(subscription)

        logger.info(
            f"Created subscription {subscription.id} for {address} on {chain.value} "
            f"(status: {subscription.status})"
        )

        return subscription

    async def unsubscribe_address(
        self,
        address: str,
        chain: ChainType,
    ) -> bool:
        """Unsubscribe from events for an address."""
        address_lower = address.lower()
        key = f"{chain.value}:{address_lower}"

        sub_id = self._address_to_subscription.get(key)
        if not sub_id:
            return False

        subscription = self._subscriptions.get(sub_id)
        if not subscription:
            return False

        # Remove from webhook
        try:
            await self._remove_address_from_webhook(
                address=address_lower,
                chain=chain,
            )
        except Exception as e:
            logger.error(f"Failed to remove address from webhook: {e}")

        # Update subscription status
        subscription.status = SubscriptionStatus.EXPIRED
        subscription.updated_at = datetime.now(timezone.utc)

        # Remove from cache
        del self._address_to_subscription[key]
        del self._subscriptions[sub_id]

        # Update in Convex
        await self._update_subscription_status(sub_id, SubscriptionStatus.EXPIRED)

        return True

    async def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get a subscription by ID."""
        return self._subscriptions.get(subscription_id)

    async def get_subscriptions_for_user(self, user_id: str) -> List[Subscription]:
        """Get all subscriptions for a user."""
        return [
            sub for sub in self._subscriptions.values()
            if sub.user_id == user_id
        ]

    async def get_watched_addresses(self, chain: Optional[ChainType] = None) -> List[str]:
        """Get all watched addresses, optionally filtered by chain."""
        addresses = []
        for sub in self._subscriptions.values():
            if sub.status == SubscriptionStatus.ACTIVE:
                if chain is None or sub.chain == chain:
                    addresses.append(sub.address)
        return addresses

    # =========================================================================
    # Webhook Handling
    # =========================================================================

    async def handle_webhook(
        self,
        provider: str,
        payload: bytes,
        signature: Optional[str] = None,
    ) -> List[WalletActivity]:
        """
        Handle incoming webhook from a provider.

        Args:
            provider: "alchemy" or "helius"
            payload: Raw webhook payload bytes
            signature: Webhook signature for verification

        Returns:
            List of processed wallet activities
        """
        handler = self._handlers.get(provider)
        if not handler:
            raise ValueError(f"Unknown webhook provider: {provider}")

        # Verify signature
        if signature and not handler.verify_signature(payload, signature):
            raise ValueError("Invalid webhook signature")

        # Parse payload
        import json
        raw_payload = json.loads(payload)
        webhook_payload = handler.parse_payload(raw_payload)

        # Get watched addresses for this chain
        watched_addresses = await self.get_watched_addresses(webhook_payload.chain)

        if not watched_addresses:
            logger.debug(f"No watched addresses for chain {webhook_payload.chain}")
            return []

        # Parse activities
        activities = await handler.parse_activities(webhook_payload, watched_addresses)

        if not activities:
            logger.debug("No relevant activities in webhook")
            return []

        # Process activities
        processed = await self._processor.process_activities(activities)

        # Update subscription last_activity_at
        for activity in processed:
            key = f"{activity.chain.value}:{activity.wallet_address}"
            sub_id = self._address_to_subscription.get(key)
            if sub_id and sub_id in self._subscriptions:
                self._subscriptions[sub_id].last_activity_at = datetime.now(timezone.utc)

        logger.info(
            f"Processed {len(processed)} activities from {provider} webhook"
        )

        return processed

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_event(
        self,
        event_type: EventType,
        callback: EventCallback,
    ):
        """Register a callback for an event type."""
        self._processor.register_callback(event_type, callback)

    def off_event(
        self,
        event_type: EventType,
        callback: EventCallback,
    ):
        """Unregister a callback."""
        self._processor.unregister_callback(event_type, callback)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _register_webhook_for_address(
        self,
        address: str,
        chain: ChainType,
    ) -> WebhookRegistration:
        """Register a webhook for an address."""
        # Determine provider based on chain
        if chain == ChainType.SOLANA:
            provider = "helius"
        else:
            provider = "alchemy"

        handler = self._handlers.get(provider)
        if not handler:
            raise ValueError(f"No handler configured for {provider}")

        # Check if we have an existing webhook for this chain
        webhook_key = f"{provider}:{chain.value}"
        existing = self._webhook_registrations.get(webhook_key)

        if existing:
            # Add address to existing webhook
            if provider == "alchemy":
                await handler.add_addresses_to_webhook(existing.webhook_id, [address])
            else:
                # Helius requires updating the full list
                all_addresses = await self.get_watched_addresses(chain)
                if address not in all_addresses:
                    all_addresses.append(address)
                await handler.update_webhook_addresses(existing.webhook_id, all_addresses)

            return existing

        # Create new webhook
        if not self.webhook_base_url:
            raise ValueError("webhook_base_url not configured")

        webhook_url = f"{self.webhook_base_url}/webhooks/{provider}"

        config = WebhookConfig(
            chain=chain,
            addresses=[address],
            webhook_url=webhook_url,
            alchemy_network=self._chain_to_alchemy_network(chain) if provider == "alchemy" else None,
        )

        registration = await handler.create_webhook(config)
        self._webhook_registrations[webhook_key] = registration

        return registration

    async def _remove_address_from_webhook(
        self,
        address: str,
        chain: ChainType,
    ):
        """Remove an address from webhook monitoring."""
        if chain == ChainType.SOLANA:
            provider = "helius"
        else:
            provider = "alchemy"

        handler = self._handlers.get(provider)
        if not handler:
            return

        webhook_key = f"{provider}:{chain.value}"
        registration = self._webhook_registrations.get(webhook_key)

        if not registration:
            return

        if provider == "alchemy":
            await handler.remove_addresses_from_webhook(registration.webhook_id, [address])
        else:
            # Helius: update with all addresses except this one
            all_addresses = await self.get_watched_addresses(chain)
            updated = [a for a in all_addresses if a != address]
            if updated:
                await handler.update_webhook_addresses(registration.webhook_id, updated)
            else:
                # No more addresses, delete webhook
                await handler.delete_webhook(registration.webhook_id)
                del self._webhook_registrations[webhook_key]

    def _chain_to_alchemy_network(self, chain: ChainType) -> str:
        """Convert chain type to Alchemy network identifier."""
        network_map = {
            ChainType.ETHEREUM: "ETH_MAINNET",
            ChainType.POLYGON: "MATIC_MAINNET",
            ChainType.ARBITRUM: "ARB_MAINNET",
            ChainType.OPTIMISM: "OPT_MAINNET",
            ChainType.BASE: "BASE_MAINNET",
            ChainType.AVALANCHE: "AVAX_MAINNET",
            ChainType.BSC: "BNB_MAINNET",
        }
        return network_map.get(chain, "ETH_MAINNET")

    async def _persist_subscription(self, subscription: Subscription):
        """Persist subscription to Convex."""
        if not self.convex_client:
            return

        try:
            await self.convex_client.mutation(
                "subscriptions:upsert",
                {
                    "id": subscription.id,
                    "userId": subscription.user_id,
                    "address": subscription.address,
                    "chain": subscription.chain.value,
                    "eventTypes": [et.value for et in subscription.event_types],
                    "webhookId": subscription.webhook_id,
                    "status": subscription.status.value,
                    "label": subscription.label,
                    "createdAt": int(subscription.created_at.timestamp() * 1000),
                    "updatedAt": int(subscription.updated_at.timestamp() * 1000),
                    "lastActivityAt": int(subscription.last_activity_at.timestamp() * 1000) if subscription.last_activity_at else None,
                    "errorMessage": subscription.error_message,
                },
            )
        except Exception as e:
            logger.error(f"Failed to persist subscription: {e}")

    async def _update_subscription_status(
        self,
        subscription_id: str,
        status: SubscriptionStatus,
    ):
        """Update subscription status in Convex."""
        if not self.convex_client:
            return

        try:
            await self.convex_client.mutation(
                "subscriptions:updateStatus",
                {
                    "id": subscription_id,
                    "status": status.value,
                    "updatedAt": int(datetime.now(timezone.utc).timestamp() * 1000),
                },
            )
        except Exception as e:
            logger.error(f"Failed to update subscription status: {e}")

    async def load_subscriptions_from_storage(self):
        """Load subscriptions from Convex on startup."""
        if not self.convex_client:
            return

        try:
            subscriptions = await self.convex_client.query(
                "subscriptions:listActive",
                {},
            )

            for sub_data in subscriptions:
                subscription = Subscription(
                    id=sub_data["id"],
                    user_id=sub_data.get("userId"),
                    address=sub_data["address"],
                    chain=ChainType(sub_data["chain"]),
                    event_types=[EventType(et) for et in sub_data.get("eventTypes", [])],
                    webhook_id=sub_data.get("webhookId"),
                    status=SubscriptionStatus(sub_data["status"]),
                    label=sub_data.get("label"),
                    created_at=datetime.fromtimestamp(sub_data["createdAt"] / 1000, tz=timezone.utc),
                    updated_at=datetime.fromtimestamp(sub_data["updatedAt"] / 1000, tz=timezone.utc),
                )

                self._subscriptions[subscription.id] = subscription
                self._address_to_subscription[f"{subscription.chain.value}:{subscription.address}"] = subscription.id

            logger.info(f"Loaded {len(subscriptions)} subscriptions from storage")

        except Exception as e:
            logger.error(f"Failed to load subscriptions: {e}")

    async def close(self):
        """Clean up resources."""
        for handler in self._handlers.values():
            await handler.close()
