"""
Strategy State Machine

Manages state transitions for strategy execution with validation,
event emission, and persistence.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from .models import (
    ExecutionContext,
    ExecutionError,
    ExecutionStep,
    InvalidTransitionError,
    StateTransition,
    StateTransitionTrigger,
    StrategyState,
)


# Type alias for state handlers
StateHandler = Callable[[ExecutionContext], Coroutine[Any, Any, Optional[StrategyState]]]
TransitionCallback = Callable[[StateTransition, ExecutionContext], Coroutine[Any, Any, None]]


class StrategyStateMachine:
    """
    Manages strategy execution state transitions.

    Features:
    - Validates transitions against allowed transition map
    - Emits events on state changes
    - Tracks state history
    - Supports state handlers for automatic transitions
    - Timeout handling for stuck states
    """

    # Define valid state transitions
    TRANSITIONS: Dict[StrategyState, Set[StrategyState]] = {
        StrategyState.IDLE: {
            StrategyState.ANALYZING,
            StrategyState.CANCELLED,
        },
        StrategyState.ANALYZING: {
            StrategyState.PLANNING,
            StrategyState.FAILED,
            StrategyState.PAUSED,
            StrategyState.CANCELLED,
        },
        StrategyState.PLANNING: {
            StrategyState.AWAITING_APPROVAL,
            StrategyState.EXECUTING,
            StrategyState.FAILED,
            StrategyState.PAUSED,
            StrategyState.CANCELLED,
        },
        StrategyState.AWAITING_APPROVAL: {
            StrategyState.EXECUTING,
            StrategyState.PAUSED,
            StrategyState.CANCELLED,
            StrategyState.FAILED,  # Timeout or rejection
        },
        StrategyState.EXECUTING: {
            StrategyState.MONITORING,
            StrategyState.EXECUTING,  # Next step
            StrategyState.COMPLETED,  # All steps done
            StrategyState.FAILED,
            StrategyState.PAUSED,
            StrategyState.CANCELLED,
        },
        StrategyState.MONITORING: {
            StrategyState.EXECUTING,  # Continue to next step
            StrategyState.COMPLETED,
            StrategyState.FAILED,
            StrategyState.PAUSED,
            StrategyState.CANCELLED,
        },
        StrategyState.PAUSED: {
            StrategyState.ANALYZING,  # Resume from beginning
            StrategyState.EXECUTING,  # Resume from current step
            StrategyState.CANCELLED,
            StrategyState.IDLE,
        },
        StrategyState.FAILED: {
            StrategyState.IDLE,       # Reset to try again
            StrategyState.ANALYZING,  # Retry from start
        },
        StrategyState.COMPLETED: {
            StrategyState.IDLE,       # Reset for new execution
        },
        StrategyState.CANCELLED: {
            StrategyState.IDLE,       # Reset for new execution
        },
    }

    # States that can timeout
    TIMEOUT_STATES: Dict[StrategyState, int] = {
        StrategyState.ANALYZING: 300,        # 5 minutes
        StrategyState.PLANNING: 120,         # 2 minutes
        StrategyState.AWAITING_APPROVAL: 3600,  # 1 hour
        StrategyState.EXECUTING: 600,        # 10 minutes per step
        StrategyState.MONITORING: 1800,      # 30 minutes for tx confirmation
    }

    def __init__(
        self,
        context: ExecutionContext,
        logger: Optional[logging.Logger] = None,
        persist_callback: Optional[Callable[[ExecutionContext], Coroutine[Any, Any, None]]] = None,
    ):
        """
        Initialize the state machine.

        Args:
            context: The execution context to manage
            logger: Optional logger
            persist_callback: Async callback to persist state changes
        """
        self.context = context
        self.logger = logger or logging.getLogger(__name__)
        self._persist_callback = persist_callback

        # State handlers - called when entering a state
        self._state_handlers: Dict[StrategyState, StateHandler] = {}

        # Transition callbacks - called after any transition
        self._transition_callbacks: List[TransitionCallback] = []

        # Timeout task
        self._timeout_task: Optional[asyncio.Task] = None

    @property
    def current_state(self) -> StrategyState:
        """Get current state."""
        return self.context.current_state

    @property
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self.context.is_terminal

    @property
    def is_active(self) -> bool:
        """Check if actively running."""
        return self.context.is_active

    def can_transition_to(self, to_state: StrategyState) -> bool:
        """Check if transition to given state is valid."""
        allowed = self.TRANSITIONS.get(self.current_state, set())
        return to_state in allowed

    def get_allowed_transitions(self) -> Set[StrategyState]:
        """Get all states we can transition to from current state."""
        return self.TRANSITIONS.get(self.current_state, set())

    async def transition_to(
        self,
        to_state: StrategyState,
        trigger: StateTransitionTrigger = StateTransitionTrigger.AUTOMATIC,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> StateTransition:
        """
        Transition to a new state.

        Args:
            to_state: Target state
            trigger: What triggered this transition
            reason: Human-readable reason
            context: Additional context data
            error_message: Error message if transitioning due to failure
            error_code: Error code if transitioning due to failure

        Returns:
            StateTransition record

        Raises:
            InvalidTransitionError: If transition is not allowed
        """
        from_state = self.current_state

        # Validate transition
        if not self.can_transition_to(to_state):
            raise InvalidTransitionError(
                from_state=from_state,
                to_state=to_state,
                message=f"Invalid transition from {from_state.value} to {to_state.value}. "
                        f"Allowed: {[s.value for s in self.get_allowed_transitions()]}",
            )

        # Cancel any pending timeout
        self._cancel_timeout()

        # Create transition record
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            reason=reason,
            context=context or {},
            error_message=error_message,
            error_code=error_code,
        )

        # Update context
        self.context.current_state = to_state
        self.context.state_entered_at = transition.timestamp
        self.context.state_history.append(transition)

        # Update timing fields
        if to_state == StrategyState.ANALYZING and not self.context.started_at:
            self.context.started_at = transition.timestamp

        if to_state in (StrategyState.COMPLETED, StrategyState.FAILED, StrategyState.CANCELLED):
            self.context.completed_at = transition.timestamp

        # Update error fields
        if error_message:
            self.context.error_message = error_message
            self.context.error_code = error_code

        self.logger.info(
            f"Strategy {self.context.execution_id}: {from_state.value} -> {to_state.value}"
            f"{f' ({reason})' if reason else ''}"
        )

        # Persist state
        await self._persist()

        # Notify callbacks
        for callback in self._transition_callbacks:
            try:
                await callback(transition, self.context)
            except Exception as e:
                self.logger.error(f"Transition callback error: {e}")

        # Start timeout for new state if applicable
        self._start_timeout_if_needed()

        # Run state handler if registered
        handler = self._state_handlers.get(to_state)
        if handler:
            try:
                next_state = await handler(self.context)
                if next_state and next_state != to_state:
                    # Handler requested another transition
                    return await self.transition_to(
                        next_state,
                        trigger=StateTransitionTrigger.AUTOMATIC,
                        reason="State handler requested transition",
                    )
            except ExecutionError as e:
                return await self.fail(e.message, e.code, e.recoverable)
            except Exception as e:
                return await self.fail(str(e), "HANDLER_ERROR", recoverable=True)

        return transition

    async def start(self) -> StateTransition:
        """Start execution from IDLE state."""
        if self.current_state != StrategyState.IDLE:
            raise InvalidTransitionError(
                from_state=self.current_state,
                to_state=StrategyState.ANALYZING,
                message="Can only start from IDLE state",
            )
        return await self.transition_to(
            StrategyState.ANALYZING,
            trigger=StateTransitionTrigger.USER_ACTION,
            reason="Execution started",
        )

    async def approve(self, approved_by: str) -> StateTransition:
        """Approve execution to continue."""
        if self.current_state != StrategyState.AWAITING_APPROVAL:
            raise InvalidTransitionError(
                from_state=self.current_state,
                to_state=StrategyState.EXECUTING,
                message="Can only approve from AWAITING_APPROVAL state",
            )

        self.context.approved_by = approved_by
        self.context.approved_at = datetime.now(timezone.utc)

        return await self.transition_to(
            StrategyState.EXECUTING,
            trigger=StateTransitionTrigger.USER_ACTION,
            reason=f"Approved by {approved_by}",
        )

    async def pause(self, reason: Optional[str] = None) -> StateTransition:
        """Pause execution."""
        if not self.can_transition_to(StrategyState.PAUSED):
            raise InvalidTransitionError(
                from_state=self.current_state,
                to_state=StrategyState.PAUSED,
                message=f"Cannot pause from {self.current_state.value} state",
            )

        return await self.transition_to(
            StrategyState.PAUSED,
            trigger=StateTransitionTrigger.USER_ACTION,
            reason=reason or "Paused by user",
        )

    async def resume(self) -> StateTransition:
        """Resume paused execution."""
        if self.current_state != StrategyState.PAUSED:
            raise InvalidTransitionError(
                from_state=self.current_state,
                to_state=StrategyState.EXECUTING,
                message="Can only resume from PAUSED state",
            )

        # Determine where to resume based on context
        if self.context.current_step and self.context.current_step.status == "executing":
            # Resume current step
            return await self.transition_to(
                StrategyState.EXECUTING,
                trigger=StateTransitionTrigger.USER_ACTION,
                reason="Resumed execution",
            )
        else:
            # Re-analyze from beginning
            return await self.transition_to(
                StrategyState.ANALYZING,
                trigger=StateTransitionTrigger.USER_ACTION,
                reason="Resumed from beginning",
            )

    async def cancel(self, reason: Optional[str] = None) -> StateTransition:
        """Cancel execution."""
        if not self.can_transition_to(StrategyState.CANCELLED):
            raise InvalidTransitionError(
                from_state=self.current_state,
                to_state=StrategyState.CANCELLED,
                message=f"Cannot cancel from {self.current_state.value} state",
            )

        return await self.transition_to(
            StrategyState.CANCELLED,
            trigger=StateTransitionTrigger.USER_ACTION,
            reason=reason or "Cancelled by user",
        )

    async def fail(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        recoverable: bool = True,
    ) -> StateTransition:
        """Transition to failed state."""
        self.context.recoverable = recoverable

        return await self.transition_to(
            StrategyState.FAILED,
            trigger=StateTransitionTrigger.ERROR,
            reason="Execution failed",
            error_message=error_message,
            error_code=error_code,
        )

    async def complete(self) -> StateTransition:
        """Mark execution as completed."""
        return await self.transition_to(
            StrategyState.COMPLETED,
            trigger=StateTransitionTrigger.AUTOMATIC,
            reason="All steps completed successfully",
        )

    async def reset(self) -> StateTransition:
        """Reset to IDLE state for new execution."""
        if self.current_state not in (
            StrategyState.COMPLETED,
            StrategyState.FAILED,
            StrategyState.CANCELLED,
        ):
            raise InvalidTransitionError(
                from_state=self.current_state,
                to_state=StrategyState.IDLE,
                message="Can only reset from terminal states",
            )

        # Clear execution state
        self.context.steps = []
        self.context.current_step_index = 0
        self.context.error_message = None
        self.context.error_code = None
        self.context.approved_by = None
        self.context.approved_at = None
        self.context.started_at = None
        self.context.completed_at = None

        return await self.transition_to(
            StrategyState.IDLE,
            trigger=StateTransitionTrigger.USER_ACTION,
            reason="Reset for new execution",
        )

    def register_state_handler(
        self,
        state: StrategyState,
        handler: StateHandler,
    ) -> None:
        """Register a handler to be called when entering a state."""
        self._state_handlers[state] = handler

    def register_transition_callback(self, callback: TransitionCallback) -> None:
        """Register a callback to be called on any transition."""
        self._transition_callbacks.append(callback)

    def add_step(self, step: ExecutionStep) -> None:
        """Add an execution step."""
        step.step_number = len(self.context.steps)
        self.context.steps.append(step)

    async def start_step(self, step_index: Optional[int] = None) -> ExecutionStep:
        """Mark a step as started."""
        idx = step_index if step_index is not None else self.context.current_step_index
        if idx >= len(self.context.steps):
            raise ExecutionError(f"Step index {idx} out of range")

        step = self.context.steps[idx]
        step.status = "executing"
        step.started_at = datetime.now(timezone.utc)
        self.context.current_step_index = idx

        await self._persist()
        return step

    async def complete_step(
        self,
        step_index: Optional[int] = None,
        output_data: Optional[Dict[str, Any]] = None,
        tx_hash: Optional[str] = None,
    ) -> ExecutionStep:
        """Mark a step as completed."""
        idx = step_index if step_index is not None else self.context.current_step_index
        if idx >= len(self.context.steps):
            raise ExecutionError(f"Step index {idx} out of range")

        step = self.context.steps[idx]
        step.status = "completed"
        step.completed_at = datetime.now(timezone.utc)
        if output_data:
            step.output_data = output_data
        if tx_hash:
            step.tx_hash = tx_hash

        await self._persist()
        return step

    async def fail_step(
        self,
        error_message: str,
        step_index: Optional[int] = None,
    ) -> ExecutionStep:
        """Mark a step as failed."""
        idx = step_index if step_index is not None else self.context.current_step_index
        if idx >= len(self.context.steps):
            raise ExecutionError(f"Step index {idx} out of range")

        step = self.context.steps[idx]
        step.status = "failed"
        step.completed_at = datetime.now(timezone.utc)
        step.error_message = error_message
        step.retry_count += 1

        await self._persist()
        return step

    async def _persist(self) -> None:
        """Persist current state."""
        if self._persist_callback:
            try:
                await self._persist_callback(self.context)
            except Exception as e:
                self.logger.error(f"Failed to persist state: {e}")

    def _start_timeout_if_needed(self) -> None:
        """Start timeout task if current state has a timeout."""
        timeout_seconds = self.TIMEOUT_STATES.get(self.current_state)
        if timeout_seconds:
            self._timeout_task = asyncio.create_task(
                self._handle_timeout(timeout_seconds)
            )

    def _cancel_timeout(self) -> None:
        """Cancel any pending timeout task."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

    async def _handle_timeout(self, seconds: int) -> None:
        """Handle state timeout."""
        try:
            await asyncio.sleep(seconds)
            # Still in same state after timeout
            self.logger.warning(
                f"Strategy {self.context.execution_id}: "
                f"Timeout in {self.current_state.value} state after {seconds}s"
            )
            await self.fail(
                f"Timeout after {seconds} seconds in {self.current_state.value} state",
                error_code="STATE_TIMEOUT",
                recoverable=True,
            )
        except asyncio.CancelledError:
            pass  # Timeout was cancelled (state changed)

    def __del__(self):
        """Cleanup timeout task on deletion."""
        self._cancel_timeout()
