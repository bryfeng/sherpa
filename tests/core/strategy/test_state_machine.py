"""
Tests for the Strategy State Machine

Tests for StrategyStateMachine, state transitions, and handlers.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta

from app.core.strategy import (
    StrategyStateMachine,
    StrategyState,
    StateTransition,
    StateTransitionTrigger,
    ExecutionContext,
    ExecutionStep,
    InvalidTransitionError,
    ExecutionError,
    StateHandlers,
    register_handlers,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def execution_context() -> ExecutionContext:
    """Basic execution context for testing."""
    return ExecutionContext(
        strategy_id="strategy-123",
        wallet_address="0x1234567890123456789012345678901234567890",
    )


@pytest.fixture
def state_machine(execution_context: ExecutionContext) -> StrategyStateMachine:
    """State machine with basic context."""
    return StrategyStateMachine(execution_context)


@pytest.fixture
def execution_with_steps(execution_context: ExecutionContext) -> ExecutionContext:
    """Execution context with steps."""
    execution_context.steps = [
        ExecutionStep(
            step_number=0,
            description="Approve token",
            action_type="approve",
            input_data={"token": "USDC", "spender": "0xrouter"},
        ),
        ExecutionStep(
            step_number=1,
            description="Swap tokens",
            action_type="swap",
            input_data={"from": "USDC", "to": "ETH", "amount": "100"},
        ),
    ]
    return execution_context


# =============================================================================
# State Machine Basic Tests
# =============================================================================

class TestStrategyStateMachine:
    """Tests for basic state machine functionality."""

    def test_initial_state_is_idle(self, state_machine: StrategyStateMachine):
        """Test that initial state is IDLE."""
        assert state_machine.current_state == StrategyState.IDLE
        assert state_machine.is_terminal is False
        assert state_machine.is_active is False

    @pytest.mark.asyncio
    async def test_start_transitions_to_analyzing(self, state_machine: StrategyStateMachine):
        """Test starting execution transitions to ANALYZING."""
        transition = await state_machine.start()

        assert state_machine.current_state == StrategyState.ANALYZING
        assert transition.from_state == StrategyState.IDLE
        assert transition.to_state == StrategyState.ANALYZING
        assert transition.trigger == StateTransitionTrigger.USER_ACTION

    @pytest.mark.asyncio
    async def test_valid_transitions(self, state_machine: StrategyStateMachine):
        """Test valid state transitions."""
        await state_machine.start()  # IDLE -> ANALYZING

        await state_machine.transition_to(StrategyState.PLANNING)
        assert state_machine.current_state == StrategyState.PLANNING

        await state_machine.transition_to(StrategyState.EXECUTING)
        assert state_machine.current_state == StrategyState.EXECUTING

        await state_machine.transition_to(StrategyState.COMPLETED)
        assert state_machine.current_state == StrategyState.COMPLETED
        assert state_machine.is_terminal is True

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, state_machine: StrategyStateMachine):
        """Test that invalid transitions raise an error."""
        # Can't go from IDLE to COMPLETED directly
        with pytest.raises(InvalidTransitionError) as exc_info:
            await state_machine.transition_to(StrategyState.COMPLETED)

        assert "idle" in str(exc_info.value).lower()
        assert "completed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_can_transition_to_check(self, state_machine: StrategyStateMachine):
        """Test can_transition_to helper."""
        assert state_machine.can_transition_to(StrategyState.ANALYZING) is True
        assert state_machine.can_transition_to(StrategyState.COMPLETED) is False

    @pytest.mark.asyncio
    async def test_get_allowed_transitions(self, state_machine: StrategyStateMachine):
        """Test getting allowed transitions."""
        allowed = state_machine.get_allowed_transitions()

        assert StrategyState.ANALYZING in allowed
        assert StrategyState.CANCELLED in allowed
        assert StrategyState.COMPLETED not in allowed


# =============================================================================
# State History Tests
# =============================================================================

class TestStateHistory:
    """Tests for state history tracking."""

    @pytest.mark.asyncio
    async def test_transitions_recorded(self, state_machine: StrategyStateMachine):
        """Test that transitions are recorded in history."""
        await state_machine.start()
        await state_machine.transition_to(StrategyState.PLANNING)

        history = state_machine.context.state_history
        assert len(history) == 2

        assert history[0].from_state == StrategyState.IDLE
        assert history[0].to_state == StrategyState.ANALYZING
        assert history[1].from_state == StrategyState.ANALYZING
        assert history[1].to_state == StrategyState.PLANNING

    @pytest.mark.asyncio
    async def test_transition_includes_reason(self, state_machine: StrategyStateMachine):
        """Test that transitions include reason."""
        await state_machine.transition_to(
            StrategyState.ANALYZING,
            reason="Starting analysis",
        )

        transition = state_machine.context.state_history[-1]
        assert transition.reason == "Starting analysis"

    @pytest.mark.asyncio
    async def test_transition_includes_context(self, state_machine: StrategyStateMachine):
        """Test that transitions include context data."""
        await state_machine.transition_to(
            StrategyState.ANALYZING,
            context={"market_data": "fetched"},
        )

        transition = state_machine.context.state_history[-1]
        assert transition.context == {"market_data": "fetched"}


# =============================================================================
# User Action Tests
# =============================================================================

class TestUserActions:
    """Tests for user actions (pause, resume, cancel, approve)."""

    @pytest.mark.asyncio
    async def test_pause_from_active_state(self, state_machine: StrategyStateMachine):
        """Test pausing from an active state."""
        await state_machine.start()  # ANALYZING
        await state_machine.pause(reason="User requested pause")

        assert state_machine.current_state == StrategyState.PAUSED

    @pytest.mark.asyncio
    async def test_resume_from_paused(self, state_machine: StrategyStateMachine):
        """Test resuming from paused state."""
        await state_machine.start()
        await state_machine.pause()
        await state_machine.resume()

        # Should go back to ANALYZING
        assert state_machine.current_state == StrategyState.ANALYZING

    @pytest.mark.asyncio
    async def test_cancel_from_active(self, state_machine: StrategyStateMachine):
        """Test cancelling from an active state."""
        await state_machine.start()
        await state_machine.cancel(reason="User cancelled")

        assert state_machine.current_state == StrategyState.CANCELLED
        assert state_machine.is_terminal is True

    @pytest.mark.asyncio
    async def test_approve_from_awaiting(self, state_machine: StrategyStateMachine):
        """Test approving from awaiting approval state."""
        await state_machine.start()
        await state_machine.transition_to(StrategyState.PLANNING)
        await state_machine.transition_to(StrategyState.AWAITING_APPROVAL)

        await state_machine.approve(approved_by="user@example.com")

        assert state_machine.current_state == StrategyState.EXECUTING
        assert state_machine.context.approved_by == "user@example.com"
        assert state_machine.context.approved_at is not None

    @pytest.mark.asyncio
    async def test_approve_not_from_awaiting_raises(self, state_machine: StrategyStateMachine):
        """Test that approving from wrong state raises."""
        await state_machine.start()

        with pytest.raises(InvalidTransitionError):
            await state_machine.approve(approved_by="user@example.com")


# =============================================================================
# Failure and Recovery Tests
# =============================================================================

class TestFailureRecovery:
    """Tests for failure handling and recovery."""

    @pytest.mark.asyncio
    async def test_fail_records_error(self, state_machine: StrategyStateMachine):
        """Test that failing records error info."""
        await state_machine.start()
        await state_machine.fail(
            error_message="Something went wrong",
            error_code="TEST_ERROR",
            recoverable=True,
        )

        assert state_machine.current_state == StrategyState.FAILED
        assert state_machine.context.error_message == "Something went wrong"
        assert state_machine.context.error_code == "TEST_ERROR"
        assert state_machine.context.recoverable is True

    @pytest.mark.asyncio
    async def test_reset_from_terminal(self, state_machine: StrategyStateMachine):
        """Test resetting from terminal state."""
        await state_machine.start()
        await state_machine.fail("Error")
        await state_machine.reset()

        assert state_machine.current_state == StrategyState.IDLE
        assert state_machine.context.error_message is None

    @pytest.mark.asyncio
    async def test_reset_not_from_active_raises(self, state_machine: StrategyStateMachine):
        """Test that resetting from active state raises."""
        await state_machine.start()

        with pytest.raises(InvalidTransitionError):
            await state_machine.reset()


# =============================================================================
# Execution Step Tests
# =============================================================================

class TestExecutionSteps:
    """Tests for execution step management."""

    @pytest.mark.asyncio
    async def test_add_step(
        self,
        execution_context: ExecutionContext,
    ):
        """Test adding steps."""
        sm = StrategyStateMachine(execution_context)

        step = ExecutionStep(description="Test step", action_type="test")
        sm.add_step(step)

        assert len(sm.context.steps) == 1
        assert sm.context.steps[0].step_number == 0

    @pytest.mark.asyncio
    async def test_start_step(
        self,
        execution_with_steps: ExecutionContext,
    ):
        """Test starting a step."""
        sm = StrategyStateMachine(execution_with_steps)

        step = await sm.start_step(0)

        assert step.status == "executing"
        assert step.started_at is not None
        assert sm.context.current_step_index == 0

    @pytest.mark.asyncio
    async def test_complete_step(
        self,
        execution_with_steps: ExecutionContext,
    ):
        """Test completing a step."""
        sm = StrategyStateMachine(execution_with_steps)

        await sm.start_step(0)
        step = await sm.complete_step(
            output_data={"tx_hash": "0x123"},
            tx_hash="0x123",
        )

        assert step.status == "completed"
        assert step.completed_at is not None
        assert step.tx_hash == "0x123"

    @pytest.mark.asyncio
    async def test_fail_step(
        self,
        execution_with_steps: ExecutionContext,
    ):
        """Test failing a step."""
        sm = StrategyStateMachine(execution_with_steps)

        await sm.start_step(0)
        step = await sm.fail_step("Transaction failed")

        assert step.status == "failed"
        assert step.error_message == "Transaction failed"
        assert step.retry_count == 1

    def test_current_step_property(self, execution_with_steps: ExecutionContext):
        """Test current step property."""
        sm = StrategyStateMachine(execution_with_steps)

        assert sm.context.current_step is not None
        assert sm.context.current_step.step_number == 0

    def test_progress_percent(self, execution_with_steps: ExecutionContext):
        """Test progress calculation."""
        sm = StrategyStateMachine(execution_with_steps)

        # No steps completed
        assert sm.context.progress_percent == 0.0

        # Complete one step
        sm.context.steps[0].status = "completed"
        assert sm.context.progress_percent == 50.0


# =============================================================================
# Timing Tests
# =============================================================================

class TestTiming:
    """Tests for timing functionality."""

    @pytest.mark.asyncio
    async def test_started_at_set_on_analyzing(self, state_machine: StrategyStateMachine):
        """Test that started_at is set when entering ANALYZING."""
        assert state_machine.context.started_at is None

        await state_machine.start()

        assert state_machine.context.started_at is not None

    @pytest.mark.asyncio
    async def test_completed_at_set_on_terminal(self, state_machine: StrategyStateMachine):
        """Test that completed_at is set on terminal states."""
        await state_machine.start()
        assert state_machine.context.completed_at is None

        await state_machine.transition_to(StrategyState.PLANNING)
        await state_machine.transition_to(StrategyState.EXECUTING)
        await state_machine.complete()

        assert state_machine.context.completed_at is not None

    def test_duration_calculation(self, execution_context: ExecutionContext):
        """Test duration calculation."""
        execution_context.started_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        execution_context.completed_at = None

        duration = execution_context.duration_seconds
        assert duration is not None
        assert duration >= 10


# =============================================================================
# Callback Tests
# =============================================================================

class TestCallbacks:
    """Tests for transition callbacks."""

    @pytest.mark.asyncio
    async def test_transition_callback_called(self, state_machine: StrategyStateMachine):
        """Test that transition callbacks are called."""
        callback_called = []

        async def callback(transition, context):
            callback_called.append(transition)

        state_machine.register_transition_callback(callback)

        await state_machine.start()

        assert len(callback_called) == 1
        assert callback_called[0].to_state == StrategyState.ANALYZING

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, state_machine: StrategyStateMachine):
        """Test multiple callbacks are all called."""
        count = [0]

        async def callback1(t, c):
            count[0] += 1

        async def callback2(t, c):
            count[0] += 10

        state_machine.register_transition_callback(callback1)
        state_machine.register_transition_callback(callback2)

        await state_machine.start()

        assert count[0] == 11


# =============================================================================
# State Handler Tests
# =============================================================================

class TestStateHandlers:
    """Tests for state handlers."""

    @pytest.mark.asyncio
    async def test_analyze_handler(self, execution_context: ExecutionContext):
        """Test analyze handler."""
        analyzed = []

        async def analyze_func(context):
            analyzed.append(context)
            return {"conditions_met": True}

        handlers = StateHandlers(analyze_func=analyze_func)
        sm = StrategyStateMachine(execution_context)
        sm.register_state_handler(StrategyState.ANALYZING, handlers.handle_analyzing)

        await sm.start()

        assert len(analyzed) == 1
        # Handler should auto-transition to PLANNING
        assert sm.current_state == StrategyState.PLANNING

    @pytest.mark.asyncio
    async def test_plan_handler_creates_steps(
        self,
        execution_context: ExecutionContext,
    ):
        """Test plan handler creates steps."""

        async def analyze_func(context):
            return {"conditions_met": True}

        async def plan_func(context, analysis):
            return [
                ExecutionStep(description="Step 1", action_type="test"),
                ExecutionStep(description="Step 2", action_type="test"),
            ]

        handlers = StateHandlers(
            analyze_func=analyze_func,
            plan_func=plan_func,
            # No execute_func - we just want to verify planning works
        )
        sm = StrategyStateMachine(execution_context)
        # Only register analyze and plan handlers (not execute)
        sm.register_state_handler(StrategyState.ANALYZING, handlers.handle_analyzing)
        sm.register_state_handler(StrategyState.PLANNING, handlers.handle_planning)

        await sm.start()

        assert len(sm.context.steps) == 2
        # Should reach EXECUTING state (handlers stop after planning)
        assert sm.current_state == StrategyState.EXECUTING

    @pytest.mark.asyncio
    async def test_plan_handler_requires_approval(
        self,
        execution_context: ExecutionContext,
    ):
        """Test plan handler requires approval for high value."""

        async def analyze_func(context):
            return {"conditions_met": True}

        async def plan_func(context, analysis):
            return [
                ExecutionStep(
                    description="Big swap",
                    action_type="swap",
                    input_data={"value_usd": 5000},
                ),
            ]

        handlers = StateHandlers(
            analyze_func=analyze_func,
            plan_func=plan_func,
            approval_threshold_usd=1000,
        )
        sm = StrategyStateMachine(execution_context)
        register_handlers(sm, handlers)

        await sm.start()

        assert sm.current_state == StrategyState.AWAITING_APPROVAL
        assert sm.context.requires_approval is True


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization."""

    def test_execution_context_to_dict(self, execution_context: ExecutionContext):
        """Test ExecutionContext serialization."""
        data = execution_context.to_dict()

        assert data["strategyId"] == "strategy-123"
        assert data["currentState"] == "idle"
        assert "isTerminal" in data
        assert "progressPercent" in data

    def test_state_transition_to_dict(self):
        """Test StateTransition serialization."""
        transition = StateTransition(
            from_state=StrategyState.IDLE,
            to_state=StrategyState.ANALYZING,
            trigger=StateTransitionTrigger.USER_ACTION,
            reason="Test",
        )

        data = transition.to_dict()

        assert data["fromState"] == "idle"
        assert data["toState"] == "analyzing"
        assert data["trigger"] == "user_action"

    def test_execution_step_to_dict(self):
        """Test ExecutionStep serialization."""
        step = ExecutionStep(
            description="Swap tokens",
            action_type="swap",
            status="completed",
            tx_hash="0x123",
        )

        data = step.to_dict()

        assert data["description"] == "Swap tokens"
        assert data["actionType"] == "swap"
        assert data["txHash"] == "0x123"
