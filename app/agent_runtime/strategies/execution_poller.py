"""
Strategy Execution Poller - Phase 13.1

Polls Convex for due strategies and creates pending execution records
that require user approval in chat.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from ..strategy import ExecutionContext, Strategy, StrategyConfig
from ...db import get_convex_client


class ExecutionPollerConfig(StrategyConfig):
    """Configuration for the execution poller strategy."""

    interval_seconds: float = Field(
        default=30.0,
        description="How often to check for due strategies (seconds).",
    )
    send_chat_notifications: bool = Field(
        default=True,
        description="Whether to send approval request messages to chat.",
    )


class StrategyExecutionPoller(Strategy):
    """
    Polls for due strategies and initiates the approval flow.

    This strategy runs periodically and:
    1. Calls checkDueStrategies mutation to find active strategies ready for execution
    2. For each new pending execution, sends an approval notification to chat
    3. The user can then approve or skip the execution via chat
    """

    id = "strategy_execution_poller"
    description = "Polls for due strategies and sends approval notifications"
    default_interval_seconds = 30.0
    ConfigModel = ExecutionPollerConfig

    def __init__(
        self,
        config: Optional[ExecutionPollerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(config=config, logger=logger)
        self._convex = None

    @property
    def convex(self):
        """Lazy-load the Convex client."""
        if self._convex is None:
            self._convex = get_convex_client()
        return self._convex

    async def on_start(self, ctx: ExecutionContext) -> None:
        ctx.logger.info(
            "StrategyExecutionPoller online; interval=%ss",
            self.interval_seconds,
        )

    async def on_tick(self, ctx: ExecutionContext) -> None:
        """Check for due strategies and create pending executions."""
        try:
            # Call existing Convex mutation to check for due strategies
            result = await self.convex.mutation(
                "strategyExecutions:checkDueStrategies",
                {},
            )

            checked = result.get("checked", 0)
            due = result.get("due", 0)
            created = result.get("created", 0)
            results = result.get("results", [])

            if due > 0:
                ctx.logger.info(
                    "Execution poller: checked=%d, due=%d, created=%d",
                    checked,
                    due,
                    created,
                )

            # Send approval notifications for newly created executions
            cfg: ExecutionPollerConfig = self.config  # type: ignore[assignment]
            if cfg.send_chat_notifications and created > 0:
                for item in results:
                    execution_id = item.get("executionId")
                    strategy_id = item.get("strategyId")
                    if execution_id:
                        await self._send_approval_notification(
                            ctx,
                            execution_id=execution_id,
                            strategy_id=strategy_id,
                        )

        except Exception as exc:
            ctx.logger.warning(
                "StrategyExecutionPoller tick failed: %s",
                exc,
                exc_info=True,
            )

    async def _send_approval_notification(
        self,
        ctx: ExecutionContext,
        execution_id: str,
        strategy_id: str,
    ) -> None:
        """
        Send an approval notification to the user's chat.

        This inserts an assistant message in the user's most recent conversation
        asking them to approve or skip the execution.
        """
        try:
            # Get execution with strategy details
            execution = await self.convex.query(
                "strategyExecutions:get",
                {"executionId": execution_id},
            )

            if not execution:
                ctx.logger.warning(
                    "Execution not found for notification: %s",
                    execution_id,
                )
                return

            wallet_address = execution.get("walletAddress")
            strategy = execution.get("strategy", {})
            approval_reason = execution.get("approvalReason", "Strategy execution ready")

            if not wallet_address:
                ctx.logger.warning(
                    "No wallet address for execution: %s",
                    execution_id,
                )
                return

            # Get or create conversation for this wallet
            conversation_id = await self._get_or_create_conversation(
                wallet_address,
                ctx,
            )

            if not conversation_id:
                ctx.logger.warning(
                    "Could not find/create conversation for wallet: %s",
                    wallet_address,
                )
                return

            # Format and send the approval message
            message = self._format_approval_message(
                strategy=strategy,
                execution_id=execution_id,
                approval_reason=approval_reason,
            )

            await self.convex.mutation(
                "conversations:addMessage",
                {
                    "conversationId": conversation_id,
                    "role": "assistant",
                    "content": message,
                    "metadata": {
                        "type": "strategy_approval_request",
                        "executionId": execution_id,
                        "strategyId": strategy_id,
                        "requiresApproval": True,
                    },
                },
            )

            ctx.logger.info(
                "Sent approval notification for execution %s to conversation %s",
                execution_id,
                conversation_id,
            )

        except Exception as exc:
            ctx.logger.warning(
                "Failed to send approval notification for %s: %s",
                execution_id,
                exc,
                exc_info=True,
            )

    async def _get_or_create_conversation(
        self,
        wallet_address: str,
        ctx: ExecutionContext,
        chain: str = "ethereum",
    ) -> Optional[str]:
        """
        Get the most recent active conversation for a wallet, or create one.

        Returns the conversation ID or None if wallet not found.
        """
        try:
            # First, find the wallet by address and chain
            wallet = await self.convex.query(
                "wallets:getByAddress",
                {"address": wallet_address.lower(), "chain": chain},
            )

            if not wallet:
                ctx.logger.debug(
                    "Wallet not found in Convex: %s",
                    wallet_address,
                )
                return None

            wallet_id = wallet.get("_id")
            if not wallet_id:
                return None

            # Get most recent non-archived conversation
            conversations = await self.convex.query(
                "conversations:listByWallet",
                {"walletId": wallet_id, "includeArchived": False},
            )

            if conversations and len(conversations) > 0:
                # Return most recent conversation
                return conversations[0].get("_id")

            # Create a new conversation if none exists
            new_conversation_id = await self.convex.mutation(
                "conversations:create",
                {
                    "walletId": wallet_id,
                    "title": "Strategy Notifications",
                },
            )

            ctx.logger.info(
                "Created new conversation for wallet %s: %s",
                wallet_address,
                new_conversation_id,
            )

            return new_conversation_id

        except Exception as exc:
            ctx.logger.warning(
                "Error getting/creating conversation for %s: %s",
                wallet_address,
                exc,
            )
            return None

    def _format_approval_message(
        self,
        strategy: Dict[str, Any],
        execution_id: str,
        approval_reason: str,
    ) -> str:
        """Format a human-readable approval request message."""
        strategy_name = strategy.get("name", "Unnamed Strategy")
        strategy_type = strategy.get("strategyType", "custom")

        return f"""**Strategy Execution Ready**

Your **{strategy_name}** ({strategy_type}) strategy is ready to execute:

> {approval_reason}

**Reply "approve" to execute, or "skip" to skip this execution.**

_Execution ID: `{execution_id}`_"""

    async def on_stop(self, ctx: ExecutionContext) -> None:
        ctx.logger.info("StrategyExecutionPoller shutting down")
