"""
Webhook Handlers

Process incoming webhooks from Alchemy (EVM) and Helius (Solana).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx

from .models import (
    ChainType,
    CHAIN_NAME_MAP,
    CHAIN_ID_MAP,
    EventType,
    ParsedTransaction,
    TokenTransfer,
    TransactionAction,
    SwapDetails,
    WalletActivity,
    WebhookPayload,
    WebhookConfig,
    WebhookRegistration,
)

logger = logging.getLogger(__name__)


class WebhookHandler(ABC):
    """Base class for webhook handlers."""

    @abstractmethod
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        pass

    @abstractmethod
    def parse_payload(self, payload: Dict[str, Any]) -> WebhookPayload:
        """Parse raw webhook payload into normalized format."""
        pass

    @abstractmethod
    async def parse_activities(
        self,
        payload: WebhookPayload,
        watched_addresses: List[str],
    ) -> List[WalletActivity]:
        """Parse webhook payload into wallet activities."""
        pass

    @abstractmethod
    async def create_webhook(self, config: WebhookConfig) -> WebhookRegistration:
        """Create a webhook subscription with the provider."""
        pass

    @abstractmethod
    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook subscription."""
        pass


class AlchemyWebhookHandler(WebhookHandler):
    """
    Handle Alchemy Address Activity webhooks.

    Alchemy provides webhooks for:
    - Address Activity: Track token transfers and native transfers
    - Mined Transactions: Track when transactions are confirmed
    - Dropped Transactions: Track failed transactions

    Docs: https://docs.alchemy.com/reference/address-activity-webhook
    """

    # Known contract addresses for classification
    KNOWN_ROUTERS: Dict[str, str] = {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3",
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "uniswap_v2",
        "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3_old",
        "0xdef1c0ded9bec7f1a1670819833240f027b25eff": "0x",
        "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch",
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "uniswap_universal",
    }

    KNOWN_PROTOCOLS: Dict[str, str] = {
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "aave_v2",
        "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "aave_v3",
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "weth",
        "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "lido",
    }

    def __init__(
        self,
        api_key: str,
        signing_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.signing_key = signing_key
        self.base_url = "https://dashboard.alchemy.com/api"
        self.client = httpx.AsyncClient(timeout=30.0)

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify Alchemy webhook signature."""
        if not self.signing_key:
            # SECURITY: Fail closed - reject unsigned webhooks in production
            logger.error("Webhook signing key not configured - rejecting webhook for security")
            return False

        expected = hmac.new(
            self.signing_key.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def parse_payload(self, payload: Dict[str, Any]) -> WebhookPayload:
        """Parse Alchemy webhook payload."""
        webhook_id = payload.get("webhookId")
        webhook_type = payload.get("type")
        network = payload.get("event", {}).get("network", "eth-mainnet")

        # Map network to chain
        chain = CHAIN_NAME_MAP.get(network, ChainType.ETHEREUM)

        # Extract events
        events = []
        event_data = payload.get("event", {})

        if webhook_type == "ADDRESS_ACTIVITY":
            activities = event_data.get("activity", [])
            events = activities
        elif webhook_type == "MINED_TRANSACTION":
            # Single transaction
            events = [event_data]
        else:
            events = [event_data]

        return WebhookPayload(
            provider="alchemy",
            chain=chain,
            webhook_id=webhook_id,
            webhook_type=webhook_type,
            events=events,
            raw=payload,
        )

    async def parse_activities(
        self,
        payload: WebhookPayload,
        watched_addresses: List[str],
    ) -> List[WalletActivity]:
        """Parse Alchemy activities into WalletActivity records."""
        activities: List[WalletActivity] = []
        watched_set = {addr.lower() for addr in watched_addresses}

        for event in payload.events:
            try:
                activity = self._parse_activity_event(event, watched_set, payload.chain)
                if activity:
                    activities.append(activity)
            except Exception as e:
                logger.error(f"Error parsing Alchemy event: {e}", exc_info=True)

        return activities

    def _parse_activity_event(
        self,
        event: Dict[str, Any],
        watched_set: set,
        chain: ChainType,
    ) -> Optional[WalletActivity]:
        """Parse a single Alchemy activity event."""
        from_addr = (event.get("fromAddress") or "").lower()
        to_addr = (event.get("toAddress") or "").lower()
        tx_hash = event.get("hash", "")

        # Determine which watched address is involved
        wallet_address = None
        direction = "internal"

        if from_addr in watched_set:
            wallet_address = from_addr
            direction = "out"
        elif to_addr in watched_set:
            wallet_address = to_addr
            direction = "in"
        else:
            # Neither address is watched
            return None

        # Parse block info
        block_num = event.get("blockNum")
        if isinstance(block_num, str) and block_num.startswith("0x"):
            block_num = int(block_num, 16)
        elif isinstance(block_num, str):
            block_num = int(block_num)

        # Parse timestamp (Alchemy may not always include it)
        timestamp = datetime.now(timezone.utc)

        # Determine event type from category
        category = event.get("category", "external")
        asset = event.get("asset", "ETH")

        if category == "external":
            event_type = EventType.NATIVE_TRANSFER_OUT if direction == "out" else EventType.NATIVE_TRANSFER_IN
        elif category == "erc20":
            event_type = EventType.TRANSFER_OUT if direction == "out" else EventType.TRANSFER_IN
        elif category == "erc721" or category == "erc1155":
            event_type = EventType.NFT_TRANSFER
        else:
            event_type = EventType.CONTRACT_CALL

        # Parse value
        value = event.get("value")
        value_usd = None
        if value is not None:
            try:
                value_usd = Decimal(str(value))
            except (TypeError, ValueError):
                pass

        # Build token transfer if applicable
        token_transfer = None
        if category in ("erc20", "erc721", "erc1155"):
            raw_contract = event.get("rawContract", {})
            token_transfer = TokenTransfer(
                token_address=raw_contract.get("address", ""),
                token_symbol=asset,
                token_decimals=raw_contract.get("decimals"),
                from_address=from_addr,
                to_address=to_addr,
                amount_raw=str(raw_contract.get("value", "0")),
                value_usd=value_usd,
            )

        # Check if this might be a swap
        is_copyable = False
        if self._looks_like_swap(event):
            event_type = EventType.SWAP
            is_copyable = True

        # Build parsed transaction
        parsed_tx = ParsedTransaction(
            tx_hash=tx_hash,
            chain=chain,
            block_number=block_num or 0,
            block_timestamp=timestamp,
            from_address=from_addr,
            to_address=to_addr,
            success=True,
            actions=[TransactionAction(type=event_type, transfer=token_transfer)],
            token_transfers=[token_transfer] if token_transfer else [],
        )

        counterparty = to_addr if direction == "out" else from_addr

        return WalletActivity(
            wallet_address=wallet_address,
            chain=chain,
            event_type=event_type,
            tx_hash=tx_hash,
            block_number=block_num or 0,
            timestamp=timestamp,
            direction=direction,
            value_usd=value_usd,
            parsed_tx=parsed_tx,
            counterparty_address=counterparty,
            counterparty_label=self._get_address_label(counterparty),
            is_copyable=is_copyable,
        )

    def _looks_like_swap(self, event: Dict[str, Any]) -> bool:
        """Heuristic to detect if an event is part of a swap."""
        to_addr = (event.get("toAddress") or "").lower()
        from_addr = (event.get("fromAddress") or "").lower()

        # Check if interacting with known DEX router
        if to_addr in self.KNOWN_ROUTERS or from_addr in self.KNOWN_ROUTERS:
            return True

        return False

    def _get_address_label(self, address: str) -> Optional[str]:
        """Get human-readable label for known addresses."""
        addr_lower = address.lower()

        if addr_lower in self.KNOWN_ROUTERS:
            return self.KNOWN_ROUTERS[addr_lower]
        if addr_lower in self.KNOWN_PROTOCOLS:
            return self.KNOWN_PROTOCOLS[addr_lower]

        return None

    async def create_webhook(self, config: WebhookConfig) -> WebhookRegistration:
        """Create an Alchemy webhook."""
        # Determine Alchemy network
        network_map = {
            ChainType.ETHEREUM: "ETH_MAINNET",
            ChainType.POLYGON: "MATIC_MAINNET",
            ChainType.ARBITRUM: "ARB_MAINNET",
            ChainType.OPTIMISM: "OPT_MAINNET",
            ChainType.BASE: "BASE_MAINNET",
        }
        network = network_map.get(config.chain, "ETH_MAINNET")

        payload = {
            "network": network,
            "webhook_type": "ADDRESS_ACTIVITY",
            "webhook_url": config.webhook_url,
            "addresses": config.addresses,
        }

        response = await self.client.post(
            f"{self.base_url}/create-webhook",
            json=payload,
            headers={
                "X-Alchemy-Token": self.api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

        return WebhookRegistration(
            provider="alchemy",
            webhook_id=data.get("id", ""),
            chain=config.chain,
            addresses=config.addresses,
            webhook_url=config.webhook_url,
            status="active",
        )

    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete an Alchemy webhook."""
        response = await self.client.delete(
            f"{self.base_url}/delete-webhook",
            params={"webhook_id": webhook_id},
            headers={"X-Alchemy-Token": self.api_key},
        )
        return response.status_code == 200

    async def add_addresses_to_webhook(
        self,
        webhook_id: str,
        addresses: List[str],
    ) -> bool:
        """Add addresses to an existing webhook."""
        response = await self.client.patch(
            f"{self.base_url}/update-webhook-addresses",
            json={
                "webhook_id": webhook_id,
                "addresses_to_add": addresses,
            },
            headers={
                "X-Alchemy-Token": self.api_key,
                "Content-Type": "application/json",
            },
        )
        return response.status_code == 200

    async def remove_addresses_from_webhook(
        self,
        webhook_id: str,
        addresses: List[str],
    ) -> bool:
        """Remove addresses from a webhook."""
        response = await self.client.patch(
            f"{self.base_url}/update-webhook-addresses",
            json={
                "webhook_id": webhook_id,
                "addresses_to_remove": addresses,
            },
            headers={
                "X-Alchemy-Token": self.api_key,
                "Content-Type": "application/json",
            },
        )
        return response.status_code == 200

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class HeliusWebhookHandler(WebhookHandler):
    """
    Handle Helius webhooks for Solana.

    Helius provides:
    - Transaction webhooks
    - Account webhooks
    - NFT events

    Docs: https://docs.helius.dev/webhooks-and-websockets/webhooks
    """

    # Known Solana programs
    KNOWN_PROGRAMS: Dict[str, str] = {
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "jupiter",
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "orca_whirlpool",
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "orca",
        "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "raydium_clmm",
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium",
        "mv3ekLzLbnVPNxjSKvqBpU3ZeZXPQdEC3bp5MDEBG68": "marinade",
        "CrX7kMhLC3cSsXJdT7JDgqrRVWGnUpX3gfEfxxU2NVLi": "solend",
    }

    def __init__(
        self,
        api_key: str,
    ):
        self.api_key = api_key
        self.base_url = "https://api.helius.xyz/v0"
        self.client = httpx.AsyncClient(timeout=30.0)

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify Helius webhook signature."""
        # Helius uses the API key as the signing secret
        expected = hmac.new(
            self.api_key.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def parse_payload(self, payload: Dict[str, Any]) -> WebhookPayload:
        """Parse Helius webhook payload."""
        # Helius sends array of transactions
        if isinstance(payload, list):
            events = payload
        else:
            events = [payload]

        return WebhookPayload(
            provider="helius",
            chain=ChainType.SOLANA,
            webhook_id=None,
            webhook_type="enhanced_transaction",
            events=events,
            raw={"events": events},
        )

    async def parse_activities(
        self,
        payload: WebhookPayload,
        watched_addresses: List[str],
    ) -> List[WalletActivity]:
        """Parse Helius events into WalletActivity records."""
        activities: List[WalletActivity] = []
        watched_set = {addr.lower() for addr in watched_addresses}

        for event in payload.events:
            try:
                parsed = self._parse_helius_transaction(event, watched_set)
                if parsed:
                    activities.extend(parsed)
            except Exception as e:
                logger.error(f"Error parsing Helius event: {e}", exc_info=True)

        return activities

    def _parse_helius_transaction(
        self,
        tx: Dict[str, Any],
        watched_set: set,
    ) -> List[WalletActivity]:
        """Parse a Helius enhanced transaction."""
        activities: List[WalletActivity] = []

        signature = tx.get("signature", "")
        slot = tx.get("slot", 0)
        timestamp_unix = tx.get("timestamp", 0)
        timestamp = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)

        tx_type = tx.get("type", "UNKNOWN")
        source = tx.get("source", "")
        fee_payer = tx.get("feePayer", "").lower()

        # Map Helius transaction types to our EventType
        type_map = {
            "SWAP": EventType.SWAP,
            "TRANSFER": EventType.TRANSFER_OUT,
            "NFT_SALE": EventType.NFT_SALE,
            "NFT_MINT": EventType.NFT_MINT,
            "STAKE_SOL": EventType.STAKE,
            "UNSTAKE_SOL": EventType.UNSTAKE,
            "UNKNOWN": EventType.UNKNOWN,
        }
        event_type = type_map.get(tx_type, EventType.CONTRACT_CALL)

        # Check native transfers
        native_transfers = tx.get("nativeTransfers", [])
        for transfer in native_transfers:
            from_addr = (transfer.get("fromUserAccount") or "").lower()
            to_addr = (transfer.get("toUserAccount") or "").lower()
            amount_lamports = transfer.get("amount", 0)

            wallet_address = None
            direction = "internal"

            if from_addr in watched_set:
                wallet_address = from_addr
                direction = "out"
            elif to_addr in watched_set:
                wallet_address = to_addr
                direction = "in"

            if wallet_address:
                activities.append(WalletActivity(
                    wallet_address=wallet_address,
                    chain=ChainType.SOLANA,
                    event_type=EventType.NATIVE_TRANSFER_OUT if direction == "out" else EventType.NATIVE_TRANSFER_IN,
                    tx_hash=signature,
                    block_number=slot,
                    timestamp=timestamp,
                    direction=direction,
                    value_usd=None,  # Would need price lookup
                    counterparty_address=to_addr if direction == "out" else from_addr,
                    is_copyable=(tx_type == "SWAP"),
                ))

        # Check token transfers
        token_transfers = tx.get("tokenTransfers", [])
        for transfer in token_transfers:
            from_addr = (transfer.get("fromUserAccount") or "").lower()
            to_addr = (transfer.get("toUserAccount") or "").lower()
            mint = transfer.get("mint", "")
            amount = transfer.get("tokenAmount", 0)

            wallet_address = None
            direction = "internal"

            if from_addr in watched_set:
                wallet_address = from_addr
                direction = "out"
            elif to_addr in watched_set:
                wallet_address = to_addr
                direction = "in"

            if wallet_address:
                token_transfer = TokenTransfer(
                    token_address=mint,
                    token_symbol=transfer.get("tokenStandard"),
                    from_address=from_addr,
                    to_address=to_addr,
                    amount_raw=str(amount),
                )

                # For swaps, detect based on tx type
                if tx_type == "SWAP":
                    event_type = EventType.SWAP
                else:
                    event_type = EventType.TRANSFER_OUT if direction == "out" else EventType.TRANSFER_IN

                activities.append(WalletActivity(
                    wallet_address=wallet_address,
                    chain=ChainType.SOLANA,
                    event_type=event_type,
                    tx_hash=signature,
                    block_number=slot,
                    timestamp=timestamp,
                    direction=direction,
                    counterparty_address=to_addr if direction == "out" else from_addr,
                    counterparty_label=self._get_program_label(source),
                    is_copyable=(tx_type == "SWAP"),
                ))

        # If no specific transfers but fee payer is watched, record contract call
        if not activities and fee_payer in watched_set:
            activities.append(WalletActivity(
                wallet_address=fee_payer,
                chain=ChainType.SOLANA,
                event_type=event_type,
                tx_hash=signature,
                block_number=slot,
                timestamp=timestamp,
                direction="out",
                counterparty_label=self._get_program_label(source),
                is_copyable=(tx_type == "SWAP"),
            ))

        return activities

    def _get_program_label(self, program_id: str) -> Optional[str]:
        """Get label for known Solana programs."""
        return self.KNOWN_PROGRAMS.get(program_id)

    async def create_webhook(self, config: WebhookConfig) -> WebhookRegistration:
        """Create a Helius webhook."""
        webhook_type = config.helius_webhook_type or "enhanced"

        payload = {
            "webhookURL": config.webhook_url,
            "transactionTypes": ["SWAP", "TRANSFER", "NFT_SALE"],
            "accountAddresses": config.addresses,
            "webhookType": webhook_type,
        }

        response = await self.client.post(
            f"{self.base_url}/webhooks",
            params={"api-key": self.api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return WebhookRegistration(
            provider="helius",
            webhook_id=data.get("webhookID", ""),
            chain=ChainType.SOLANA,
            addresses=config.addresses,
            webhook_url=config.webhook_url,
            status="active",
        )

    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a Helius webhook."""
        response = await self.client.delete(
            f"{self.base_url}/webhooks/{webhook_id}",
            params={"api-key": self.api_key},
        )
        return response.status_code == 200

    async def update_webhook_addresses(
        self,
        webhook_id: str,
        addresses: List[str],
    ) -> bool:
        """Update addresses on a Helius webhook."""
        response = await self.client.put(
            f"{self.base_url}/webhooks/{webhook_id}",
            params={"api-key": self.api_key},
            json={"accountAddresses": addresses},
        )
        return response.status_code == 200

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
