"""
Event Monitoring Service

Provides unified event monitoring across chains and protocols.
Handles webhooks from Alchemy (EVM) and Helius (Solana).
"""

from .models import (
    EventType,
    ChainType,
    WalletActivity,
    ParsedTransaction,
    TransactionAction,
    Subscription,
    SubscriptionStatus,
    WebhookPayload,
)
from .webhook_handler import WebhookHandler, AlchemyWebhookHandler, HeliusWebhookHandler
from .event_processor import EventProcessor
from .service import EventMonitoringService, get_event_monitoring_service

__all__ = [
    # Models
    "EventType",
    "ChainType",
    "WalletActivity",
    "ParsedTransaction",
    "TransactionAction",
    "Subscription",
    "SubscriptionStatus",
    "WebhookPayload",
    # Handlers
    "WebhookHandler",
    "AlchemyWebhookHandler",
    "HeliusWebhookHandler",
    # Processor
    "EventProcessor",
    # Service
    "EventMonitoringService",
    "get_event_monitoring_service",
]
