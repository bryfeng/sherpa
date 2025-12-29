"""
Strategy State Machine Handlers

Provides state-specific handlers for strategy execution.
These handlers are called when the state machine enters a specific state.
"""

import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from .models import (
    ExecutionContext,
    ExecutionError,
    ExecutionStep,
    StrategyState,
)


# Type aliases
AnalyzeFunc = Callable[[ExecutionContext], Coroutine[Any, Any, Dict[str, Any]]]
PlanFunc = Callable[[ExecutionContext, Dict[str, Any]], Coroutine[Any, Any, list[ExecutionStep]]]
ExecuteFunc = Callable[[ExecutionContext, ExecutionStep], Coroutine[Any, Any, Dict[str, Any]]]
MonitorFunc = Callable[[ExecutionContext, str], Coroutine[Any, Any, bool]]


class StateHandlers:
    """
    Collection of state handlers for strategy execution.

    These handlers implement the actual logic for each state:
    - ANALYZING: Gather market data, evaluate conditions
    - PLANNING: Determine actions to take
    - EXECUTING: Execute transactions
    - MONITORING: Wait for confirmations

    Usage:
        handlers = StateHandlers(
            analyze_func=my_analyze,
            plan_func=my_plan,
            execute_func=my_execute,
            monitor_func=my_monitor,
        )

        state_machine.register_state_handler(
            StrategyState.ANALYZING,
            handlers.handle_analyzing,
        )
    """

    def __init__(
        self,
        analyze_func: Optional[AnalyzeFunc] = None,
        plan_func: Optional[PlanFunc] = None,
        execute_func: Optional[ExecuteFunc] = None,
        monitor_func: Optional[MonitorFunc] = None,
        approval_threshold_usd: float = 1000.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize state handlers.

        Args:
            analyze_func: Function to analyze market conditions
            plan_func: Function to create execution plan
            execute_func: Function to execute a single step
            monitor_func: Function to monitor transaction status
            approval_threshold_usd: USD threshold requiring approval
            logger: Optional logger
        """
        self._analyze = analyze_func
        self._plan = plan_func
        self._execute = execute_func
        self._monitor = monitor_func
        self._approval_threshold = approval_threshold_usd
        self.logger = logger or logging.getLogger(__name__)

        # Store analysis results between states
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

    async def handle_analyzing(
        self,
        context: ExecutionContext,
    ) -> Optional[StrategyState]:
        """
        Handle ANALYZING state.

        Gathers market data and evaluates strategy conditions.
        Returns the next state to transition to.
        """
        self.logger.info(f"Analyzing strategy {context.strategy_id}")

        if not self._analyze:
            raise ExecutionError(
                "No analyze function configured",
                code="MISSING_HANDLER",
                recoverable=False,
            )

        try:
            # Run analysis
            analysis = await self._analyze(context)
            self._analysis_cache[context.execution_id] = analysis

            # Check if conditions are met
            if not analysis.get("conditions_met", True):
                self.logger.info(
                    f"Strategy conditions not met: {analysis.get('reason', 'unknown')}"
                )
                return StrategyState.COMPLETED  # Skip execution

            # Proceed to planning
            return StrategyState.PLANNING

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise ExecutionError(str(e), code="ANALYSIS_FAILED", recoverable=True)

    async def handle_planning(
        self,
        context: ExecutionContext,
    ) -> Optional[StrategyState]:
        """
        Handle PLANNING state.

        Creates the execution plan (list of steps) based on analysis.
        Returns the next state to transition to.
        """
        self.logger.info(f"Planning execution for strategy {context.strategy_id}")

        if not self._plan:
            raise ExecutionError(
                "No plan function configured",
                code="MISSING_HANDLER",
                recoverable=False,
            )

        try:
            # Get cached analysis
            analysis = self._analysis_cache.get(context.execution_id, {})

            # Create execution plan
            steps = await self._plan(context, analysis)

            if not steps:
                self.logger.info("No steps to execute, completing")
                return StrategyState.COMPLETED

            # Add steps to context
            context.steps = steps
            context.current_step_index = 0

            # Calculate total value for approval check
            total_value_usd = sum(
                step.input_data.get("value_usd", 0) for step in steps
            )

            # Check if approval required
            if total_value_usd > self._approval_threshold:
                context.requires_approval = True
                context.approval_reason = (
                    f"Total value ${total_value_usd:.2f} exceeds "
                    f"threshold ${self._approval_threshold:.2f}"
                )
                return StrategyState.AWAITING_APPROVAL

            # Proceed to execution
            return StrategyState.EXECUTING

        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            raise ExecutionError(str(e), code="PLANNING_FAILED", recoverable=True)

    async def handle_executing(
        self,
        context: ExecutionContext,
    ) -> Optional[StrategyState]:
        """
        Handle EXECUTING state.

        Executes the current step in the plan.
        Returns the next state to transition to.
        """
        step = context.current_step
        if not step:
            # No more steps, complete
            return StrategyState.COMPLETED

        self.logger.info(
            f"Executing step {step.step_number + 1}/{len(context.steps)}: "
            f"{step.description}"
        )

        if not self._execute:
            raise ExecutionError(
                "No execute function configured",
                code="MISSING_HANDLER",
                recoverable=False,
            )

        try:
            # Mark step as executing
            step.status = "executing"

            # Execute the step
            result = await self._execute(context, step)

            # Check result
            if result.get("success"):
                step.status = "completed"
                step.output_data = result.get("output", {})
                step.tx_hash = result.get("tx_hash")

                # If there's a transaction, monitor it
                if step.tx_hash:
                    return StrategyState.MONITORING

                # Otherwise move to next step
                return self._advance_step(context)

            else:
                step.status = "failed"
                step.error_message = result.get("error", "Unknown error")
                step.retry_count += 1

                # Check if we should retry
                if step.retry_count < 3 and result.get("recoverable", True):
                    self.logger.warning(
                        f"Step failed, retrying ({step.retry_count}/3): "
                        f"{step.error_message}"
                    )
                    step.status = "pending"
                    return StrategyState.EXECUTING  # Retry

                raise ExecutionError(
                    step.error_message or "Step failed",
                    code="STEP_FAILED",
                    recoverable=False,
                )

        except ExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            raise ExecutionError(str(e), code="EXECUTION_FAILED", recoverable=True)

    async def handle_monitoring(
        self,
        context: ExecutionContext,
    ) -> Optional[StrategyState]:
        """
        Handle MONITORING state.

        Waits for transaction confirmation.
        Returns the next state to transition to.
        """
        step = context.current_step
        if not step or not step.tx_hash:
            return self._advance_step(context)

        self.logger.info(f"Monitoring transaction: {step.tx_hash}")

        if not self._monitor:
            # No monitor function, assume success
            self.logger.warning("No monitor function, assuming transaction succeeded")
            step.status = "completed"
            return self._advance_step(context)

        try:
            # Check transaction status
            confirmed = await self._monitor(context, step.tx_hash)

            if confirmed:
                step.status = "completed"
                return self._advance_step(context)
            else:
                # Still pending, stay in monitoring
                return None

        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
            raise ExecutionError(str(e), code="MONITORING_FAILED", recoverable=True)

    def _advance_step(self, context: ExecutionContext) -> StrategyState:
        """Advance to the next step or complete."""
        context.current_step_index += 1

        if context.current_step_index >= len(context.steps):
            # All steps done
            return StrategyState.COMPLETED

        # More steps to execute
        return StrategyState.EXECUTING

    def clear_cache(self, execution_id: str) -> None:
        """Clear cached analysis for an execution."""
        self._analysis_cache.pop(execution_id, None)


def register_handlers(
    state_machine: Any,  # StrategyStateMachine
    handlers: StateHandlers,
) -> None:
    """
    Register all state handlers with a state machine.

    Usage:
        handlers = StateHandlers(...)
        register_handlers(state_machine, handlers)
    """
    state_machine.register_state_handler(
        StrategyState.ANALYZING,
        handlers.handle_analyzing,
    )
    state_machine.register_state_handler(
        StrategyState.PLANNING,
        handlers.handle_planning,
    )
    state_machine.register_state_handler(
        StrategyState.EXECUTING,
        handlers.handle_executing,
    )
    state_machine.register_state_handler(
        StrategyState.MONITORING,
        handlers.handle_monitoring,
    )
