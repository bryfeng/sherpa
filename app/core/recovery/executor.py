"""
Recovery Executor

High-level executor that combines error classification, retry logic,
and escalation for resilient operation execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

from .errors import (
    ErrorCategory,
    ErrorContext,
    RecoverableError,
    UnrecoverableError,
    classify_error,
)
from .strategies import (
    CircuitBreakerConfig,
    CircuitBreakerStrategy,
    ExponentialBackoffStrategy,
    RetryConfig,
)

T = TypeVar("T")


@dataclass
class ExecutionResult:
    """Result of an operation execution."""

    success: bool
    result: Optional[Any] = None
    error: Optional[Exception] = None
    error_context: Optional[ErrorContext] = None

    # Execution metadata
    attempts: int = 1
    total_duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Recovery info
    recovered: bool = False
    recovery_strategy_used: Optional[str] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "error": str(self.error) if self.error else None,
            "errorContext": self.error_context.__dict__ if self.error_context else None,
            "attempts": self.attempts,
            "totalDurationSeconds": self.total_duration_seconds,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "recovered": self.recovered,
            "recoveryStrategyUsed": self.recovery_strategy_used,
            "escalated": self.escalated,
            "escalationReason": self.escalation_reason,
        }


@dataclass
class RecoveryConfig:
    """Configuration for the recovery executor."""

    # Retry settings
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0

    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: float = 60.0

    # Escalation settings
    escalate_on_failure: bool = True
    max_escalation_delay_seconds: float = 300.0

    # Category-specific settings
    category_configs: Dict[ErrorCategory, RetryConfig] = field(default_factory=dict)


# Type for escalation callback
EscalationCallback = Callable[[Exception, ErrorContext, int], Coroutine[Any, Any, None]]


class RecoveryExecutor:
    """
    Executes operations with comprehensive error recovery.

    Features:
    - Automatic error classification
    - Exponential backoff retry
    - Circuit breaker protection
    - Escalation to humans for unrecoverable errors
    - Detailed execution metrics
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        escalation_callback: Optional[EscalationCallback] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or RecoveryConfig()
        self.escalation_callback = escalation_callback
        self.logger = logger or logging.getLogger(__name__)

        # Circuit breakers per provider
        self._circuit_breakers: Dict[str, CircuitBreakerStrategy] = {}

        # Build retry strategy
        self._retry_strategy = ExponentialBackoffStrategy(
            max_attempts=self.config.max_retries,
            initial_delay=self.config.initial_delay_seconds,
            max_delay=self.config.max_delay_seconds,
            exponential_base=self.config.exponential_base,
            logger=self.logger,
        )

    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        operation_name: str = "operation",
        provider: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute an operation with full recovery support.

        Args:
            operation: Async operation to execute
            operation_name: Name for logging
            provider: Provider name (for circuit breaker)
            context: Additional context for logging

        Returns:
            ExecutionResult with success/failure info
        """
        started_at = datetime.now(timezone.utc)
        attempts = 0
        last_error: Optional[Exception] = None
        last_error_context: Optional[ErrorContext] = None

        try:
            # Apply circuit breaker if enabled and provider specified
            if self.config.enable_circuit_breaker and provider:
                circuit = self._get_circuit_breaker(provider)
                operation = self._wrap_with_circuit_breaker(operation, circuit)

            # Execute with retry
            for attempt in range(self.config.max_retries):
                attempts = attempt + 1

                try:
                    result = await operation()

                    return ExecutionResult(
                        success=True,
                        result=result,
                        attempts=attempts,
                        total_duration_seconds=self._duration_seconds(started_at),
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        recovered=attempts > 1,
                        recovery_strategy_used="retry" if attempts > 1 else None,
                    )

                except UnrecoverableError as e:
                    # Don't retry unrecoverable errors
                    last_error = e
                    last_error_context = e.context
                    self.logger.error(
                        f"{operation_name} failed with unrecoverable error: {e}"
                    )
                    break

                except RecoverableError as e:
                    last_error = e
                    last_error_context = e.context

                    if attempt < self.config.max_retries - 1:
                        delay = self._get_retry_delay(e, attempt)
                        self.logger.warning(
                            f"{operation_name} attempt {attempts}/{self.config.max_retries} "
                            f"failed: {e}. Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        self.logger.error(
                            f"{operation_name} failed after {attempts} attempts: {e}"
                        )

                except Exception as e:
                    # Classify unknown error
                    error_ctx = classify_error(e)
                    last_error = e
                    last_error_context = error_ctx

                    if not error_ctx.recoverable:
                        self.logger.error(
                            f"{operation_name} failed with classified unrecoverable error: {e}"
                        )
                        break

                    if attempt < self.config.max_retries - 1:
                        delay = error_ctx.retry_after_seconds or self.config.initial_delay_seconds
                        self.logger.warning(
                            f"{operation_name} attempt {attempts}/{self.config.max_retries} "
                            f"failed: {e}. Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        self.logger.error(
                            f"{operation_name} failed after {attempts} attempts: {e}"
                        )

            # All retries exhausted or unrecoverable error
            escalated = False
            escalation_reason = None

            if self.config.escalate_on_failure and self.escalation_callback:
                try:
                    await self.escalation_callback(
                        last_error,
                        last_error_context or ErrorContext(),
                        attempts,
                    )
                    escalated = True
                    escalation_reason = "All retries exhausted"
                except Exception as esc_error:
                    self.logger.error(f"Escalation failed: {esc_error}")

            return ExecutionResult(
                success=False,
                error=last_error,
                error_context=last_error_context,
                attempts=attempts,
                total_duration_seconds=self._duration_seconds(started_at),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                recovered=False,
                escalated=escalated,
                escalation_reason=escalation_reason,
            )

        except Exception as e:
            # Unexpected error in executor itself
            self.logger.exception(f"Unexpected error in recovery executor: {e}")
            return ExecutionResult(
                success=False,
                error=e,
                error_context=ErrorContext(
                    category=ErrorCategory.UNKNOWN,
                    recoverable=False,
                ),
                attempts=attempts,
                total_duration_seconds=self._duration_seconds(started_at),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

    async def execute_with_fallbacks(
        self,
        primary: Callable[[], Coroutine[Any, Any, T]],
        fallbacks: List[Callable[[], Coroutine[Any, Any, T]]],
        operation_name: str = "operation",
    ) -> ExecutionResult:
        """
        Execute with fallback operations.

        If primary fails after retries, tries fallbacks in order.
        """
        started_at = datetime.now(timezone.utc)
        total_attempts = 0
        all_errors: List[Exception] = []

        # Try primary
        result = await self.execute(primary, f"{operation_name} (primary)")
        total_attempts += result.attempts

        if result.success:
            return result

        all_errors.append(result.error)

        # Try fallbacks
        for i, fallback in enumerate(fallbacks):
            self.logger.info(
                f"Primary failed, trying fallback {i + 1}/{len(fallbacks)}"
            )

            result = await self.execute(
                fallback, f"{operation_name} (fallback {i + 1})"
            )
            total_attempts += result.attempts

            if result.success:
                return ExecutionResult(
                    success=True,
                    result=result.result,
                    attempts=total_attempts,
                    total_duration_seconds=self._duration_seconds(started_at),
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    recovered=True,
                    recovery_strategy_used=f"fallback_{i + 1}",
                )

            all_errors.append(result.error)

        # All failed
        return ExecutionResult(
            success=False,
            error=UnrecoverableError(
                f"All operations failed: {[str(e) for e in all_errors]}"
            ),
            error_context=ErrorContext(
                category=ErrorCategory.UNKNOWN,
                recoverable=False,
                details={"errors": [str(e) for e in all_errors]},
            ),
            attempts=total_attempts,
            total_duration_seconds=self._duration_seconds(started_at),
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            escalated=result.escalated,
            escalation_reason="All fallbacks exhausted",
        )

    def get_circuit_breaker_state(self, provider: str) -> Optional[str]:
        """Get the state of a circuit breaker."""
        circuit = self._circuit_breakers.get(provider)
        return circuit.state.value if circuit else None

    def reset_circuit_breaker(self, provider: str) -> bool:
        """Reset a circuit breaker manually."""
        circuit = self._circuit_breakers.get(provider)
        if circuit:
            circuit.reset()
            return True
        return False

    def _get_circuit_breaker(self, provider: str) -> CircuitBreakerStrategy:
        """Get or create a circuit breaker for a provider."""
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreakerStrategy(
                name=provider,
                config=CircuitBreakerConfig(
                    failure_threshold=self.config.circuit_failure_threshold,
                    timeout_seconds=self.config.circuit_timeout_seconds,
                ),
                logger=self.logger,
            )
        return self._circuit_breakers[provider]

    def _wrap_with_circuit_breaker(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        circuit: CircuitBreakerStrategy,
    ) -> Callable[[], Coroutine[Any, Any, T]]:
        """Wrap an operation with circuit breaker."""

        async def wrapped() -> T:
            return await circuit.execute(operation)

        return wrapped

    def _get_retry_delay(self, error: Exception, attempt: int) -> float:
        """Calculate retry delay based on error and attempt."""
        if isinstance(error, RecoverableError) and error.retry_after:
            return error.retry_after

        # Exponential backoff with jitter
        import random

        delay = min(
            self.config.initial_delay_seconds * (self.config.exponential_base ** attempt),
            self.config.max_delay_seconds,
        )
        jitter = delay * 0.1
        return delay + random.uniform(-jitter, jitter)

    def _duration_seconds(self, started_at: datetime) -> float:
        """Calculate duration since start."""
        return (datetime.now(timezone.utc) - started_at).total_seconds()


# Convenience function for simple usage
async def execute_with_recovery(
    operation: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    operation_name: str = "operation",
) -> ExecutionResult:
    """
    Execute an operation with default recovery settings.

    Simple wrapper for quick usage without configuring RecoveryExecutor.
    """
    executor = RecoveryExecutor(
        config=RecoveryConfig(max_retries=max_retries)
    )
    return await executor.execute(operation, operation_name)
