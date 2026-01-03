"""
Tests for Event Monitoring Service

Tests webhook handling, event processing, and subscription management.
"""

import json
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.events.models import (
    ChainType,
    EventType,
    WalletActivity,
    ParsedTransaction,
    TokenTransfer,
    Subscription,
    SubscriptionStatus,
    WebhookPayload,
)
from app.services.events.webhook_handler import (
    AlchemyWebhookHandler,
    HeliusWebhookHandler,
)
from app.services.events.event_processor import (
    EventProcessor,
    CopyTradingEventFilter,
)


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Test event monitoring models."""

    def test_event_type_values(self):
        """Test EventType enum has expected values."""
        assert EventType.SWAP == "swap"
        assert EventType.TRANSFER_IN == "transfer_in"
        assert EventType.TRANSFER_OUT == "transfer_out"
        assert EventType.BRIDGE_INITIATE == "bridge_initiate"
        assert EventType.DEPOSIT == "deposit"
        assert EventType.STAKE == "stake"

    def test_chain_type_values(self):
        """Test ChainType enum has expected values."""
        assert ChainType.ETHEREUM == "ethereum"
        assert ChainType.POLYGON == "polygon"
        assert ChainType.SOLANA == "solana"
        assert ChainType.ARBITRUM == "arbitrum"
        assert ChainType.BASE == "base"

    def test_wallet_activity_creation(self):
        """Test WalletActivity model creation."""
        activity = WalletActivity(
            wallet_address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xabc123",
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            value_usd=Decimal("100.50"),
            is_copyable=True,
            copy_relevance_score=0.8,
        )

        assert activity.wallet_address == "0x1234567890123456789012345678901234567890"
        assert activity.chain == ChainType.ETHEREUM
        assert activity.event_type == EventType.SWAP
        assert activity.is_copyable is True
        assert activity.copy_relevance_score == 0.8

    def test_subscription_creation(self):
        """Test Subscription model creation."""
        sub = Subscription(
            address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_types=[EventType.SWAP, EventType.TRANSFER_IN],
            label="Test Wallet",
        )

        assert sub.address == "0x1234567890123456789012345678901234567890"
        assert sub.chain == ChainType.ETHEREUM
        assert len(sub.event_types) == 2
        assert sub.status == SubscriptionStatus.PENDING
        assert sub.is_active() is False

    def test_token_transfer_native_detection(self):
        """Test native token detection."""
        native = TokenTransfer(
            token_address="0x0000000000000000000000000000000000000000",
            from_address="0x1111",
            to_address="0x2222",
            amount_raw="1000000000000000000",
        )
        assert native.is_native is True

        erc20 = TokenTransfer(
            token_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
            from_address="0x1111",
            to_address="0x2222",
            amount_raw="1000000",
        )
        assert erc20.is_native is False


# =============================================================================
# Alchemy Webhook Handler Tests
# =============================================================================


class TestAlchemyWebhookHandler:
    """Test Alchemy webhook handling."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return AlchemyWebhookHandler(
            api_key="test_key",
            signing_key="test_signing_key",
        )

    def test_parse_payload(self, handler):
        """Test parsing Alchemy webhook payload."""
        raw_payload = {
            "webhookId": "wh_123",
            "type": "ADDRESS_ACTIVITY",
            "event": {
                "network": "eth-mainnet",
                "activity": [
                    {
                        "hash": "0xabc123",
                        "fromAddress": "0x1111111111111111111111111111111111111111",
                        "toAddress": "0x2222222222222222222222222222222222222222",
                        "blockNum": "0x1234",
                        "category": "erc20",
                        "asset": "USDC",
                        "value": 100.5,
                        "rawContract": {
                            "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                            "value": "100500000",
                            "decimals": 6,
                        },
                    }
                ],
            },
        }

        payload = handler.parse_payload(raw_payload)

        assert payload.provider == "alchemy"
        assert payload.chain == ChainType.ETHEREUM
        assert payload.webhook_id == "wh_123"
        assert payload.webhook_type == "ADDRESS_ACTIVITY"
        assert len(payload.events) == 1

    @pytest.mark.asyncio
    async def test_parse_activities(self, handler):
        """Test parsing activities from webhook payload."""
        payload = WebhookPayload(
            provider="alchemy",
            chain=ChainType.ETHEREUM,
            webhook_id="wh_123",
            webhook_type="ADDRESS_ACTIVITY",
            events=[
                {
                    "hash": "0xabc123",
                    "fromAddress": "0x1111111111111111111111111111111111111111",
                    "toAddress": "0x2222222222222222222222222222222222222222",
                    "blockNum": 12345,
                    "category": "erc20",
                    "asset": "USDC",
                    "value": 100.5,
                    "rawContract": {
                        "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                        "value": "100500000",
                        "decimals": 6,
                    },
                }
            ],
            raw={},
        )

        watched = ["0x1111111111111111111111111111111111111111"]
        activities = await handler.parse_activities(payload, watched)

        assert len(activities) == 1
        activity = activities[0]
        assert activity.wallet_address == "0x1111111111111111111111111111111111111111"
        assert activity.direction == "out"
        assert activity.event_type == EventType.TRANSFER_OUT
        assert activity.tx_hash == "0xabc123"

    def test_swap_detection(self, handler):
        """Test swap detection via known routers."""
        # Event interacting with Uniswap V3 router
        swap_event = {
            "hash": "0xabc123",
            "fromAddress": "0x1111111111111111111111111111111111111111",
            "toAddress": "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3
            "blockNum": 12345,
            "category": "erc20",
        }

        is_swap = handler._looks_like_swap(swap_event)
        assert is_swap is True

        # Regular transfer (not a swap)
        transfer_event = {
            "hash": "0xdef456",
            "fromAddress": "0x1111111111111111111111111111111111111111",
            "toAddress": "0x2222222222222222222222222222222222222222",
            "blockNum": 12345,
            "category": "erc20",
        }

        is_swap = handler._looks_like_swap(transfer_event)
        assert is_swap is False

    def test_verify_signature(self, handler):
        """Test signature verification."""
        payload = b'{"test": "data"}'

        # Generate expected signature
        import hashlib
        import hmac

        expected = hmac.new(
            b"test_signing_key",
            payload,
            hashlib.sha256,
        ).hexdigest()

        assert handler.verify_signature(payload, expected) is True
        assert handler.verify_signature(payload, "wrong_signature") is False


# =============================================================================
# Helius Webhook Handler Tests
# =============================================================================


class TestHeliusWebhookHandler:
    """Test Helius webhook handling."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return HeliusWebhookHandler(api_key="test_key")

    def test_parse_payload(self, handler):
        """Test parsing Helius webhook payload."""
        raw_payload = [
            {
                "signature": "abc123signature",
                "slot": 12345678,
                "timestamp": 1700000000,
                "type": "SWAP",
                "source": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
                "feePayer": "SoLanaWaLLet1111111111111111111111111111111",
                "nativeTransfers": [],
                "tokenTransfers": [
                    {
                        "fromUserAccount": "SoLanaWaLLet1111111111111111111111111111111",
                        "toUserAccount": "JupiterPoOL1111111111111111111111111111111",
                        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "tokenAmount": 100,
                    }
                ],
            }
        ]

        payload = handler.parse_payload(raw_payload)

        assert payload.provider == "helius"
        assert payload.chain == ChainType.SOLANA
        assert len(payload.events) == 1

    @pytest.mark.asyncio
    async def test_parse_activities(self, handler):
        """Test parsing Helius transactions into activities."""
        payload = WebhookPayload(
            provider="helius",
            chain=ChainType.SOLANA,
            events=[
                {
                    "signature": "abc123signature",
                    "slot": 12345678,
                    "timestamp": 1700000000,
                    "type": "SWAP",
                    "source": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
                    "feePayer": "solanawallet111111111111111111111111111111",
                    "nativeTransfers": [],
                    "tokenTransfers": [
                        {
                            "fromUserAccount": "solanawallet111111111111111111111111111111",
                            "toUserAccount": "jupiterpool111111111111111111111111111111",
                            "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                            "tokenAmount": 100,
                        }
                    ],
                }
            ],
            raw={},
        )

        watched = ["solanawallet111111111111111111111111111111"]
        activities = await handler.parse_activities(payload, watched)

        assert len(activities) >= 1
        activity = activities[0]
        assert activity.chain == ChainType.SOLANA
        assert activity.event_type == EventType.SWAP
        assert activity.is_copyable is True

    def test_program_label_lookup(self, handler):
        """Test Solana program label lookup."""
        jupiter = handler._get_program_label("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
        assert jupiter == "jupiter"

        raydium = handler._get_program_label("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
        assert raydium == "raydium"

        unknown = handler._get_program_label("unknown111111111111111111111111111111111")
        assert unknown is None


# =============================================================================
# Event Processor Tests
# =============================================================================


class TestEventProcessor:
    """Test event processing and classification."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return EventProcessor()

    @pytest.mark.asyncio
    async def test_process_activities_dedup(self, processor):
        """Test deduplication of activities."""
        activity = WalletActivity(
            wallet_address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xabc123",
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            direction="out",
        )

        # Process first time
        result1 = await processor.process_activities([activity])
        assert len(result1) == 1

        # Process duplicate
        result2 = await processor.process_activities([activity])
        assert len(result2) == 0  # Should be deduplicated

    @pytest.mark.asyncio
    async def test_classify_swap_event(self, processor):
        """Test swap classification based on DEX router."""
        activity = WalletActivity(
            wallet_address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_type=EventType.TRANSFER_OUT,  # Initially classified as transfer
            tx_hash="0xabc123",
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            counterparty_address="0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3
        )

        enriched = processor._classify_event_type(activity)

        assert enriched.event_type == EventType.SWAP
        assert enriched.counterparty_label == "uniswap_v3"

    @pytest.mark.asyncio
    async def test_assess_copyability(self, processor):
        """Test copyability assessment."""
        swap_activity = WalletActivity(
            wallet_address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xabc123",
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            value_usd=Decimal("5000"),
        )

        assessed = processor._assess_copyability(swap_activity)

        assert assessed.is_copyable is True
        assert assessed.copy_relevance_score > 0.5

        # Transfer should not be copyable by default
        transfer_activity = WalletActivity(
            wallet_address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_type=EventType.TRANSFER_OUT,
            tx_hash="0xdef456",
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            direction="out",
        )

        assessed = processor._assess_copyability(transfer_activity)

        assert assessed.is_copyable is False
        assert assessed.copy_relevance_score == 0.0

    @pytest.mark.asyncio
    async def test_callback_routing(self, processor):
        """Test event routing to callbacks."""
        callback_called = []

        async def swap_callback(activity: WalletActivity):
            callback_called.append(activity)

        processor.register_callback(EventType.SWAP, swap_callback)

        activity = WalletActivity(
            wallet_address="0x1234567890123456789012345678901234567890",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xabc123",
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            direction="out",
        )

        await processor._route_to_callbacks(activity)

        assert len(callback_called) == 1
        assert callback_called[0].tx_hash == "0xabc123"


# =============================================================================
# Copy Trading Filter Tests
# =============================================================================


class TestCopyTradingEventFilter:
    """Test copy trading event filtering."""

    def test_filter_by_value(self):
        """Test filtering by USD value."""
        filter = CopyTradingEventFilter(
            min_value_usd=Decimal("50"),
            max_value_usd=Decimal("10000"),
        )

        small_trade = WalletActivity(
            wallet_address="0x1234",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xsmall",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            value_usd=Decimal("10"),  # Too small
            is_copyable=True,
        )

        assert filter.should_copy(small_trade) is False

        medium_trade = WalletActivity(
            wallet_address="0x1234",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xmedium",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            value_usd=Decimal("500"),  # In range
            is_copyable=True,
        )

        assert filter.should_copy(medium_trade) is True

        large_trade = WalletActivity(
            wallet_address="0x1234",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xlarge",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            value_usd=Decimal("50000"),  # Too large
            is_copyable=True,
        )

        assert filter.should_copy(large_trade) is False

    def test_filter_by_event_type(self):
        """Test filtering by event type."""
        filter = CopyTradingEventFilter(
            allowed_types={EventType.SWAP, EventType.BRIDGE_INITIATE},
        )

        swap = WalletActivity(
            wallet_address="0x1234",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xswap",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            is_copyable=True,
        )

        assert filter.should_copy(swap) is True

        transfer = WalletActivity(
            wallet_address="0x1234",
            chain=ChainType.ETHEREUM,
            event_type=EventType.TRANSFER_OUT,
            tx_hash="0xtransfer",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            is_copyable=True,  # Even if marked copyable, not in allowed types
        )

        assert filter.should_copy(transfer) is False

    def test_filter_by_blocked_tokens(self):
        """Test filtering by blocked tokens."""
        filter = CopyTradingEventFilter(
            blocked_tokens={"0xbadtoken111111111111111111111111111111"},
        )

        activity = WalletActivity(
            wallet_address="0x1234",
            chain=ChainType.ETHEREUM,
            event_type=EventType.SWAP,
            tx_hash="0xbad",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            direction="out",
            is_copyable=True,
            parsed_tx=ParsedTransaction(
                tx_hash="0xbad",
                chain=ChainType.ETHEREUM,
                block_number=1,
                block_timestamp=datetime.now(timezone.utc),
                from_address="0x1234",
                to_address="0x5678",
                token_transfers=[
                    TokenTransfer(
                        token_address="0xbadtoken111111111111111111111111111111",
                        from_address="0x1234",
                        to_address="0x5678",
                        amount_raw="1000",
                    )
                ],
            ),
        )

        assert filter.should_copy(activity) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestEventMonitoringIntegration:
    """Integration tests for event monitoring."""

    @pytest.mark.asyncio
    async def test_end_to_end_alchemy_webhook(self):
        """Test end-to-end Alchemy webhook processing."""
        handler = AlchemyWebhookHandler(api_key="test", signing_key=None)
        processor = EventProcessor()

        # Simulated Alchemy webhook payload
        raw_payload = {
            "webhookId": "wh_123",
            "type": "ADDRESS_ACTIVITY",
            "event": {
                "network": "eth-mainnet",
                "activity": [
                    {
                        "hash": "0xabc123",
                        "fromAddress": "0x1111111111111111111111111111111111111111",
                        "toAddress": "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap
                        "blockNum": 12345,
                        "category": "erc20",
                        "asset": "USDC",
                        "value": 500.0,
                    }
                ],
            },
        }

        # Parse webhook
        webhook_payload = handler.parse_payload(raw_payload)
        assert webhook_payload.chain == ChainType.ETHEREUM

        # Parse activities
        watched = ["0x1111111111111111111111111111111111111111"]
        activities = await handler.parse_activities(webhook_payload, watched)
        assert len(activities) == 1

        # Process activities
        processed = await processor.process_activities(activities)
        assert len(processed) == 1
        assert processed[0].event_type == EventType.SWAP
        assert processed[0].is_copyable is True

    @pytest.mark.asyncio
    async def test_end_to_end_helius_webhook(self):
        """Test end-to-end Helius webhook processing."""
        handler = HeliusWebhookHandler(api_key="test")
        processor = EventProcessor()

        # Simulated Helius webhook payload
        raw_payload = [
            {
                "signature": "abc123",
                "slot": 12345678,
                "timestamp": 1700000000,
                "type": "SWAP",
                "source": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
                "feePayer": "walletabc111111111111111111111111111111111",
                "nativeTransfers": [],
                "tokenTransfers": [
                    {
                        "fromUserAccount": "walletabc111111111111111111111111111111111",
                        "toUserAccount": "jupiterpool11111111111111111111111111111",
                        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "tokenAmount": 100,
                    }
                ],
            }
        ]

        # Parse webhook
        webhook_payload = handler.parse_payload(raw_payload)
        assert webhook_payload.chain == ChainType.SOLANA

        # Parse activities
        watched = ["walletabc111111111111111111111111111111111"]
        activities = await handler.parse_activities(webhook_payload, watched)
        assert len(activities) >= 1

        # Process activities
        processed = await processor.process_activities(activities)
        assert len(processed) >= 1

        swap_activity = next((a for a in processed if a.event_type == EventType.SWAP), None)
        assert swap_activity is not None
        assert swap_activity.is_copyable is True
