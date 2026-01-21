"""
Copy Trading Event Bridge

Bridges event monitoring to copy trading.
Listens for swap events and triggers copy trades.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, TYPE_CHECKING

from .manager import CopyTradingManager
from .models import TradeSignal, TradeAction

if TYPE_CHECKING:
    from ...services.events.models import WalletActivity, ParsedTransaction

logger = logging.getLogger(__name__)


class CopyTradingEventBridge:
    """
    Bridges event monitoring to copy trading.

    Listens for swap events from watched leader wallets and
    triggers copy trades through the CopyTradingManager.
    """

    def __init__(
        self,
        copy_trading_manager: CopyTradingManager,
        event_monitoring_service: Optional[Any] = None,
    ):
        self.manager = copy_trading_manager
        self.event_service = event_monitoring_service
        self._registered = False

    async def start(self):
        """Start listening for events."""
        if self._registered or not self.event_service:
            return

        # Import here to avoid circular imports
        from ...services.events.models import EventType

        # Register callbacks for swap events
        self.event_service.on_event(EventType.SWAP, self._handle_swap_event)
        self.event_service.on_event(EventType.TRANSFER_OUT, self._handle_transfer_event)

        self._registered = True
        logger.info("CopyTradingEventBridge started listening for events")

    async def stop(self):
        """Stop listening for events."""
        if not self._registered or not self.event_service:
            return

        from ...services.events.models import EventType

        self.event_service.off_event(EventType.SWAP, self._handle_swap_event)
        self.event_service.off_event(EventType.TRANSFER_OUT, self._handle_transfer_event)

        self._registered = False
        logger.info("CopyTradingEventBridge stopped")

    async def _handle_swap_event(self, activity: Any):
        """
        Handle a swap event from a watched wallet.

        Converts the activity to a TradeSignal and forwards to the manager.
        """
        try:
            # Check if this wallet is being followed
            leader_address = activity.wallet_address
            leader_chain = self._activity_chain_to_string(activity.chain)

            # Get relationships following this leader
            relationships = await self.manager.get_relationships_for_leader(
                leader_address=leader_address,
                leader_chain=leader_chain,
            )

            if not relationships:
                logger.debug(f"No followers for leader {leader_address}")
                return

            # Convert to TradeSignal
            signal = self._activity_to_signal(activity)
            if not signal:
                logger.warning(f"Could not convert activity to signal: {activity.id}")
                return

            # Forward to manager
            executions = await self.manager.handle_trade_signal(signal)

            logger.info(
                f"Processed swap from {leader_address}: "
                f"{len(executions)} copy executions triggered"
            )

        except Exception as e:
            logger.error(f"Error handling swap event: {e}", exc_info=True)

    async def _handle_transfer_event(self, activity: Any):
        """
        Handle a transfer event from a watched wallet.

        May be used for tracking portfolio changes of leaders.
        Currently not triggering copy trades for transfers.
        """
        # For now, just log - transfers don't trigger copies
        logger.debug(
            f"Transfer event from {activity.wallet_address}: "
            f"{activity.direction} {activity.value_usd} USD"
        )

    def _activity_to_signal(self, activity: "WalletActivity") -> Optional[TradeSignal]:
        """Convert a WalletActivity to a TradeSignal."""
        try:
            parsed_tx = activity.parsed_tx
            if not parsed_tx:
                return None

            # Extract swap details from parsed transaction actions
            swap_details = None
            token_in_address = ""
            token_in_symbol = ""
            token_out_address = ""
            token_out_symbol = ""
            token_in_amount: Optional[Decimal] = None
            token_out_amount: Optional[Decimal] = None
            dex_name = None

            # Look for swap action in parsed transaction
            if parsed_tx.actions:
                for action in parsed_tx.actions:
                    if action.swap:
                        swap_details = action.swap
                        break

            if swap_details:
                # Extract from swap details
                token_in_address = swap_details.token_in.token_address
                token_in_symbol = swap_details.token_in.token_symbol or "UNKNOWN"
                token_out_address = swap_details.token_out.token_address
                token_out_symbol = swap_details.token_out.token_symbol or "UNKNOWN"
                token_in_amount = swap_details.token_in.amount
                token_out_amount = swap_details.token_out.amount
                dex_name = swap_details.dex_name

            # Fallback to token transfers if no swap details
            elif parsed_tx.token_transfers and len(parsed_tx.token_transfers) >= 2:
                transfers = parsed_tx.token_transfers
                # First transfer is usually input, last is output
                token_in = transfers[0]
                token_out = transfers[-1]
                token_in_address = token_in.token_address
                token_in_symbol = token_in.token_symbol or "UNKNOWN"
                token_out_address = token_out.token_address
                token_out_symbol = token_out.token_symbol or "UNKNOWN"
                token_in_amount = token_in.amount
                token_out_amount = token_out.amount

            # If we still don't have token info, try to infer from transfers
            if not token_in_address or not token_out_address:
                return self._infer_signal_from_transfers(activity)

            # Handle timestamp - activity.timestamp is a datetime object
            timestamp = activity.timestamp
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

            return TradeSignal(
                leader_address=activity.wallet_address,
                leader_chain=self._activity_chain_to_string(activity.chain),
                tx_hash=activity.tx_hash,
                block_number=activity.block_number,
                timestamp=timestamp,
                action=TradeAction.SWAP,
                token_in_address=token_in_address,
                token_in_symbol=token_in_symbol,
                token_in_amount=token_in_amount,
                token_out_address=token_out_address,
                token_out_symbol=token_out_symbol,
                token_out_amount=token_out_amount,
                value_usd=activity.value_usd,
                dex=dex_name or activity.counterparty_label,
                dex_name=dex_name or activity.counterparty_label,
                raw_data=parsed_tx.model_dump() if parsed_tx else None,
            )

        except Exception as e:
            logger.error(f"Error converting activity to signal: {e}", exc_info=True)
            return None

    def _infer_signal_from_transfers(self, activity: "WalletActivity") -> Optional[TradeSignal]:
        """Infer a swap signal from token transfers."""
        parsed_tx = activity.parsed_tx
        if not parsed_tx or not parsed_tx.token_transfers:
            return None

        transfers = parsed_tx.token_transfers
        if len(transfers) < 2:
            return None

        # Find the token in (sent by user) and token out (received by user)
        token_in = None
        token_out = None

        for transfer in transfers:
            if transfer.from_address.lower() == activity.wallet_address.lower():
                token_in = transfer
            elif transfer.to_address.lower() == activity.wallet_address.lower():
                token_out = transfer

        if not token_in or not token_out:
            return None

        # Handle timestamp
        timestamp = activity.timestamp
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

        return TradeSignal(
            leader_address=activity.wallet_address,
            leader_chain=self._activity_chain_to_string(activity.chain),
            tx_hash=activity.tx_hash,
            block_number=activity.block_number,
            timestamp=timestamp,
            action=TradeAction.SWAP,
            token_in_address=token_in.token_address,
            token_in_symbol=token_in.token_symbol,
            token_in_amount=token_in.amount,
            token_out_address=token_out.token_address,
            token_out_symbol=token_out.token_symbol,
            token_out_amount=token_out.amount,
            value_usd=activity.value_usd,
            dex=activity.counterparty_label,
            raw_data=parsed_tx.model_dump() if parsed_tx else None,
        )

    def _activity_chain_to_string(self, chain: Any) -> str:
        """Convert ChainType enum to string."""
        if hasattr(chain, "value"):
            return chain.value
        return str(chain)


# Singleton instance
_bridge_instance: Optional[CopyTradingEventBridge] = None


def get_copy_trading_bridge() -> CopyTradingEventBridge:
    """Get the singleton CopyTradingEventBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        from .manager import CopyTradingManager
        from .executor import CopyExecutor
        from ...db import get_convex_client
        from ...services.events.service import get_event_monitoring_service

        convex = get_convex_client()

        # Create executor with real providers (Jupiter + Relay)
        executor = CopyExecutor.create_with_providers()

        manager = CopyTradingManager(
            convex_client=convex,
            executor=executor,
        )

        event_service = get_event_monitoring_service()

        _bridge_instance = CopyTradingEventBridge(
            copy_trading_manager=manager,
            event_monitoring_service=event_service,
        )

    return _bridge_instance


async def start_copy_trading_bridge():
    """Start the copy trading event bridge."""
    bridge = get_copy_trading_bridge()
    await bridge.start()


async def stop_copy_trading_bridge():
    """Stop the copy trading event bridge."""
    global _bridge_instance
    if _bridge_instance:
        await _bridge_instance.stop()
        _bridge_instance = None
