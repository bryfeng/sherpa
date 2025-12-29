"""
Strategy State Machine Persistence

Provides persistence callbacks for the state machine using Convex.
"""

import logging
from typing import Any, Dict, Optional

from .models import ExecutionContext, ExecutionStep, StateTransition, StrategyState


class ExecutionPersistence:
    """
    Persistence layer for strategy execution state.

    Uses Convex to persist execution state, enabling:
    - Recovery after crashes
    - State inspection from admin panel
    - Audit trail of all transitions
    """

    def __init__(
        self,
        convex_client: Any,  # Type: ConvexClient from app.db.convex_client
        logger: Optional[logging.Logger] = None,
    ):
        self.convex = convex_client
        self.logger = logger or logging.getLogger(__name__)

    async def create_execution(
        self,
        strategy_id: str,
        wallet_address: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new execution record in Convex.

        Returns the Convex execution ID.
        """
        result = await self.convex.mutation(
            "executions:create",
            {
                "strategyId": strategy_id,
                "walletAddress": wallet_address,
                "metadata": metadata,
            },
        )
        return result

    async def persist_state(self, context: ExecutionContext) -> None:
        """
        Persist execution state to Convex.

        This is called by the state machine after every state transition.
        """
        try:
            # Get the latest transition
            latest_transition = (
                context.state_history[-1] if context.state_history else None
            )

            if latest_transition:
                await self.convex.mutation(
                    "executions:updateState",
                    {
                        "executionId": context.execution_id,
                        "currentState": context.current_state.value,
                        "stateEnteredAt": int(context.state_entered_at.timestamp() * 1000),
                        "transition": self._transition_to_convex(latest_transition),
                        "startedAt": (
                            int(context.started_at.timestamp() * 1000)
                            if context.started_at
                            else None
                        ),
                        "completedAt": (
                            int(context.completed_at.timestamp() * 1000)
                            if context.completed_at
                            else None
                        ),
                        "errorMessage": context.error_message,
                        "errorCode": context.error_code,
                        "recoverable": context.recoverable,
                    },
                )
            else:
                # Full sync if no transition (e.g., step updates)
                await self.sync_full_state(context)

        except Exception as e:
            self.logger.error(f"Failed to persist state: {e}")
            raise

    async def persist_steps(self, context: ExecutionContext) -> None:
        """Persist execution steps to Convex."""
        try:
            await self.convex.mutation(
                "executions:updateSteps",
                {
                    "executionId": context.execution_id,
                    "steps": [self._step_to_convex(s) for s in context.steps],
                    "currentStepIndex": context.current_step_index,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to persist steps: {e}")
            raise

    async def persist_approval(self, context: ExecutionContext) -> None:
        """Persist approval info to Convex."""
        try:
            await self.convex.mutation(
                "executions:setApproval",
                {
                    "executionId": context.execution_id,
                    "requiresApproval": context.requires_approval,
                    "approvalReason": context.approval_reason,
                    "approvedBy": context.approved_by,
                    "approvedAt": (
                        int(context.approved_at.timestamp() * 1000)
                        if context.approved_at
                        else None
                    ),
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to persist approval: {e}")
            raise

    async def sync_full_state(self, context: ExecutionContext) -> None:
        """Full state sync to Convex (for recovery or debugging)."""
        try:
            await self.convex.mutation(
                "executions:syncState",
                {
                    "executionId": context.execution_id,
                    "currentState": context.current_state.value,
                    "stateEnteredAt": int(context.state_entered_at.timestamp() * 1000),
                    "steps": [self._step_to_convex(s) for s in context.steps],
                    "currentStepIndex": context.current_step_index,
                    "stateHistory": [
                        self._transition_to_convex(t) for t in context.state_history
                    ],
                    "startedAt": (
                        int(context.started_at.timestamp() * 1000)
                        if context.started_at
                        else None
                    ),
                    "completedAt": (
                        int(context.completed_at.timestamp() * 1000)
                        if context.completed_at
                        else None
                    ),
                    "requiresApproval": context.requires_approval,
                    "approvalReason": context.approval_reason,
                    "approvedBy": context.approved_by,
                    "approvedAt": (
                        int(context.approved_at.timestamp() * 1000)
                        if context.approved_at
                        else None
                    ),
                    "errorMessage": context.error_message,
                    "errorCode": context.error_code,
                    "recoverable": context.recoverable,
                    "metadata": context.metadata,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to sync full state: {e}")
            raise

    async def load_execution(self, execution_id: str) -> Optional[ExecutionContext]:
        """Load execution state from Convex."""
        try:
            data = await self.convex.query(
                "executions:get",
                {"executionId": execution_id},
            )
            if not data:
                return None
            return self._convex_to_context(data)
        except Exception as e:
            self.logger.error(f"Failed to load execution: {e}")
            return None

    async def add_decision(
        self,
        execution_id: str,
        decision_type: str,
        input_context: Dict[str, Any],
        reasoning: str,
        action_taken: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> str:
        """Add an agent decision to the execution audit trail."""
        try:
            return await self.convex.mutation(
                "executions:addDecision",
                {
                    "executionId": execution_id,
                    "decisionType": decision_type,
                    "inputContext": input_context,
                    "reasoning": reasoning,
                    "actionTaken": action_taken,
                    "riskAssessment": risk_assessment,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to add decision: {e}")
            raise

    def _transition_to_convex(self, transition: StateTransition) -> Dict[str, Any]:
        """Convert StateTransition to Convex format."""
        return {
            "id": transition.id,
            "fromState": transition.from_state.value,
            "toState": transition.to_state.value,
            "trigger": transition.trigger.value,
            "timestamp": int(transition.timestamp.timestamp() * 1000),
            "reason": transition.reason,
            "context": transition.context,
            "errorMessage": transition.error_message,
            "errorCode": transition.error_code,
        }

    def _step_to_convex(self, step: ExecutionStep) -> Dict[str, Any]:
        """Convert ExecutionStep to Convex format."""
        return {
            "id": step.id,
            "stepNumber": step.step_number,
            "description": step.description,
            "actionType": step.action_type,
            "status": step.status,
            "startedAt": (
                int(step.started_at.timestamp() * 1000) if step.started_at else None
            ),
            "completedAt": (
                int(step.completed_at.timestamp() * 1000) if step.completed_at else None
            ),
            "inputData": step.input_data,
            "outputData": step.output_data,
            "txHash": step.tx_hash,
            "chainId": step.chain_id,
            "gasUsed": step.gas_used,
            "gasPriceGwei": step.gas_price_gwei,
            "errorMessage": step.error_message,
            "retryCount": step.retry_count,
        }

    def _convex_to_context(self, data: Dict[str, Any]) -> ExecutionContext:
        """Convert Convex data to ExecutionContext."""
        from datetime import datetime, timezone

        def from_timestamp(ts: Optional[int]) -> Optional[datetime]:
            if ts is None:
                return None
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

        context = ExecutionContext(
            execution_id=data["_id"],
            strategy_id=data["strategyId"],
            wallet_address=data["walletAddress"],
            current_state=StrategyState(data["currentState"]),
            state_entered_at=from_timestamp(data["stateEnteredAt"]) or datetime.now(timezone.utc),
            current_step_index=data.get("currentStepIndex", 0),
            created_at=from_timestamp(data["createdAt"]) or datetime.now(timezone.utc),
            started_at=from_timestamp(data.get("startedAt")),
            completed_at=from_timestamp(data.get("completedAt")),
            requires_approval=data.get("requiresApproval", False),
            approval_reason=data.get("approvalReason"),
            approved_by=data.get("approvedBy"),
            approved_at=from_timestamp(data.get("approvedAt")),
            error_message=data.get("errorMessage"),
            error_code=data.get("errorCode"),
            recoverable=data.get("recoverable", True),
            metadata=data.get("metadata", {}),
        )

        # Convert steps
        for step_data in data.get("steps", []):
            step = ExecutionStep(
                id=step_data["id"],
                step_number=step_data["stepNumber"],
                description=step_data["description"],
                action_type=step_data["actionType"],
                status=step_data["status"],
                started_at=from_timestamp(step_data.get("startedAt")),
                completed_at=from_timestamp(step_data.get("completedAt")),
                input_data=step_data.get("inputData", {}),
                output_data=step_data.get("outputData", {}),
                tx_hash=step_data.get("txHash"),
                chain_id=step_data.get("chainId"),
                gas_used=step_data.get("gasUsed"),
                gas_price_gwei=step_data.get("gasPriceGwei"),
                error_message=step_data.get("errorMessage"),
                retry_count=step_data.get("retryCount", 0),
            )
            context.steps.append(step)

        # Convert state history
        for trans_data in data.get("stateHistory", []):
            from .models import StateTransitionTrigger

            transition = StateTransition(
                id=trans_data["id"],
                from_state=StrategyState(trans_data["fromState"]),
                to_state=StrategyState(trans_data["toState"]),
                trigger=StateTransitionTrigger(trans_data["trigger"]),
                timestamp=from_timestamp(trans_data["timestamp"]) or datetime.now(timezone.utc),
                reason=trans_data.get("reason"),
                context=trans_data.get("context", {}),
                error_message=trans_data.get("errorMessage"),
                error_code=trans_data.get("errorCode"),
            )
            context.state_history.append(transition)

        return context


def create_persistence_callback(persistence: ExecutionPersistence):
    """Create a persistence callback for the state machine."""

    async def callback(context: ExecutionContext) -> None:
        await persistence.persist_state(context)

    return callback
