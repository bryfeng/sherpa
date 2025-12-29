"""
Recovery Strategies

Defines different strategies for handling errors and retries.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

from .errors import ErrorCategory, ErrorContext, RecoverableError, UnrecoverableError

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Category-specific overrides
    category_overrides: Dict[ErrorCategory, "RetryConfig"] = field(default_factory=dict)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = min(
            self.initial_delay_seconds * (self.exponential_base ** attempt),
            self.max_delay_seconds,
        )
        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
        return max(delay, 0)


class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""

    @abstractmethod
    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Execute an operation with this recovery strategy."""
        pass

    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if the operation should be retried."""
        pass


class RetryStrategy(RecoveryStrategy):
    """
    Simple retry strategy with configurable attempts.

    Retries recoverable errors up to max_attempts times.
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or RetryConfig()
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_attempts):
            try:
                return await operation()
            except Exception as e:
                last_error = e

                if not self.should_retry(e, attempt):
                    raise

                if attempt < self.config.max_attempts - 1:
                    delay = self._get_delay(e, attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        # All attempts exhausted
        raise last_error or RuntimeError("All retry attempts exhausted")

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.config.max_attempts - 1:
            return False

        if isinstance(error, UnrecoverableError):
            return False

        if isinstance(error, RecoverableError):
            return True

        # Classify unknown errors
        from .errors import classify_error

        ctx = classify_error(error)
        return ctx.recoverable

    def _get_delay(self, error: Exception, attempt: int) -> float:
        # Check if error has a specific retry_after
        if isinstance(error, RecoverableError) and error.retry_after:
            return error.retry_after

        # Check category-specific config
        if isinstance(error, (RecoverableError, UnrecoverableError)):
            category = error.category
            if category in self.config.category_overrides:
                return self.config.category_overrides[category].get_delay(attempt)

        return self.config.get_delay(attempt)


class ExponentialBackoffStrategy(RetryStrategy):
    """
    Retry strategy with exponential backoff.

    Increases delay exponentially between retries to avoid
    overwhelming failing services.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        logger: Optional[logging.Logger] = None,
    ):
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay_seconds=initial_delay,
            max_delay_seconds=max_delay,
            exponential_base=exponential_base,
            jitter=True,
        )
        super().__init__(config, logger)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 3      # Successes to close from half-open
    timeout_seconds: float = 60.0   # Time before attempting recovery
    half_open_max_calls: int = 3    # Max calls in half-open state


class CircuitBreakerStrategy(RecoveryStrategy):
    """
    Circuit breaker pattern for preventing cascading failures.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Rejecting all requests after too many failures
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logger or logging.getLogger(__name__)

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        # Check if we should attempt recovery
        if self._state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self._transition_to_half_open()
            else:
                raise RecoverableError(
                    f"Circuit breaker '{self.name}' is open",
                    category=ErrorCategory.PROVIDER,
                    retry_after=self._time_until_recovery(),
                )

        # Check half-open call limit
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise RecoverableError(
                    f"Circuit breaker '{self.name}' half-open limit reached",
                    category=ErrorCategory.PROVIDER,
                )
            self._half_open_calls += 1

        try:
            result = await operation()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def should_retry(self, error: Exception, attempt: int) -> bool:
        # Circuit breaker doesn't handle retries itself
        return False

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self.logger.info(f"Circuit breaker '{self.name}' manually reset")

    def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self._failure_count = 0

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self._failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def _should_attempt_recovery(self) -> bool:
        if not self._last_failure_time:
            return True
        return (time.time() - self._last_failure_time) >= self.config.timeout_seconds

    def _time_until_recovery(self) -> float:
        if not self._last_failure_time:
            return 0
        elapsed = time.time() - self._last_failure_time
        return max(0, self.config.timeout_seconds - elapsed)

    def _transition_to_open(self) -> None:
        self._state = CircuitState.OPEN
        self.logger.warning(
            f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
        )

    def _transition_to_half_open(self) -> None:
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0
        self.logger.info(f"Circuit breaker '{self.name}' entering half-open state")

    def _transition_to_closed(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self.logger.info(f"Circuit breaker '{self.name}' closed, service recovered")


class FallbackStrategy(RecoveryStrategy):
    """
    Fallback strategy that tries alternative operations.

    If the primary operation fails, tries fallback operations in order.
    """

    def __init__(
        self,
        fallbacks: List[Callable[[], Coroutine[Any, Any, T]]],
        logger: Optional[logging.Logger] = None,
    ):
        self.fallbacks = fallbacks
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        errors: List[Exception] = []

        # Try primary operation
        try:
            return await operation()
        except Exception as e:
            errors.append(e)
            self.logger.warning(f"Primary operation failed: {e}")

        # Try fallbacks
        for i, fallback in enumerate(self.fallbacks):
            try:
                self.logger.info(f"Trying fallback {i + 1}/{len(self.fallbacks)}")
                return await fallback()
            except Exception as e:
                errors.append(e)
                self.logger.warning(f"Fallback {i + 1} failed: {e}")

        # All failed
        raise UnrecoverableError(
            f"All operations failed: {[str(e) for e in errors]}",
            category=ErrorCategory.UNKNOWN,
        )

    def should_retry(self, error: Exception, attempt: int) -> bool:
        # Fallback strategy handles alternatives, not retries
        return False


class CompositeStrategy(RecoveryStrategy):
    """
    Combines multiple recovery strategies.

    Applies strategies in order (e.g., circuit breaker + retry + fallback).
    """

    def __init__(
        self,
        strategies: List[RecoveryStrategy],
        logger: Optional[logging.Logger] = None,
    ):
        self.strategies = strategies
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        # Wrap operation with each strategy from innermost to outermost
        wrapped = operation
        for strategy in reversed(self.strategies):
            current_wrapped = wrapped

            async def make_wrapped(s=strategy, op=current_wrapped):
                return await s.execute(op, context)

            wrapped = make_wrapped

        return await wrapped()

    def should_retry(self, error: Exception, attempt: int) -> bool:
        return any(s.should_retry(error, attempt) for s in self.strategies)
